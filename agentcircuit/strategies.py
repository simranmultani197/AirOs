"""
Pre-built repair and retry strategies for AgentCircuit.

This module provides configurable strategies for handling different
types of errors in AI agent execution.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union
from enum import Enum
import time
import random
import json
import re

from pydantic import BaseModel

from .errors import ClassifiedError, ErrorCategory, ErrorClassifier


class RetryStrategy(Enum):
    """Available retry strategies."""
    NONE = "none"
    IMMEDIATE = "immediate"
    LINEAR_BACKOFF = "linear_backoff"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    JITTERED_BACKOFF = "jittered_backoff"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter_factor: float = 0.1  # 10% jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number."""
        if self.strategy == RetryStrategy.NONE:
            return 0.0
        elif self.strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * attempt
        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        elif self.strategy == RetryStrategy.JITTERED_BACKOFF:
            base = self.base_delay * (self.exponential_base ** (attempt - 1))
            jitter = base * self.jitter_factor * random.random()
            delay = base + jitter
        else:
            delay = self.base_delay

        return min(delay, self.max_delay)


class RepairStrategy(ABC):
    """Base class for repair strategies."""

    name: str = "base"
    description: str = "Base repair strategy"

    @abstractmethod
    def can_handle(self, error: ClassifiedError) -> bool:
        """Check if this strategy can handle the given error."""
        pass

    @abstractmethod
    def repair(
        self,
        error: ClassifiedError,
        input_state: Any,
        raw_output: Any,
        schema: Optional[Type[BaseModel]] = None,
        llm_callable: Optional[Callable[[str], str]] = None
    ) -> Any:
        """
        Attempt to repair the error.

        Args:
            error: The classified error
            input_state: Original input state
            raw_output: The failed output (if any)
            schema: Target Pydantic schema
            llm_callable: Optional LLM for repair

        Returns:
            Repaired output

        Raises:
            Exception if repair fails
        """
        pass


class JSONRepairStrategy(RepairStrategy):
    """Strategy for repairing JSON parsing errors."""

    name = "json_repair"
    description = "Repairs malformed JSON outputs"

    def can_handle(self, error: ClassifiedError) -> bool:
        # Only handle JSON parse errors, not schema validation
        # Schema validation errors should be handled by SchemaRepairStrategy
        return error.category == ErrorCategory.JSON_PARSE

    def repair(
        self,
        error: ClassifiedError,
        input_state: Any,
        raw_output: Any,
        schema: Optional[Type[BaseModel]] = None,
        llm_callable: Optional[Callable[[str], str]] = None
    ) -> Any:
        """Attempt to repair JSON."""
        if raw_output is None:
            if llm_callable:
                return self._llm_repair(error, input_state, raw_output, schema, llm_callable)
            raise ValueError("No output to repair and no LLM available")

        raw_str = str(raw_output)

        # Try common JSON fixes
        fixed = self._try_common_fixes(raw_str)
        if fixed:
            return fixed

        # Try extracting JSON from markdown
        extracted = self._extract_from_markdown(raw_str)
        if extracted:
            return extracted

        # Fall back to LLM repair
        if llm_callable:
            return self._llm_repair(error, input_state, raw_output, schema, llm_callable)

        raise ValueError(f"Could not repair JSON: {raw_str[:100]}")

    def _try_common_fixes(self, raw: str) -> Optional[Dict]:
        """Try common JSON fixes."""
        fixes = [
            raw,  # Try as-is first
            raw.strip(),
            raw.replace("'", '"'),  # Single to double quotes
            raw.replace("None", "null"),  # Python None to JSON null
            raw.replace("True", "true").replace("False", "false"),  # Python bool to JSON
        ]

        for attempt in fixes:
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue

        return None

    def _extract_from_markdown(self, raw: str) -> Optional[Dict]:
        """Extract JSON from markdown code blocks."""
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
            r"\{[\s\S]*\}",  # Raw JSON object
        ]

        for pattern in patterns:
            match = re.search(pattern, raw)
            if match:
                try:
                    content = match.group(1) if "```" in pattern else match.group(0)
                    return json.loads(content.strip())
                except (json.JSONDecodeError, IndexError):
                    continue

        return None

    def _llm_repair(
        self,
        error: ClassifiedError,
        input_state: Any,
        raw_output: Any,
        schema: Optional[Type[BaseModel]],
        llm_callable: Callable[[str], str]
    ) -> Dict:
        """Use LLM to repair JSON."""
        schema_text = ""
        if schema:
            try:
                schema_text = json.dumps(schema.model_json_schema(), indent=2)
            except AttributeError:
                schema_text = str(schema)

        prompt = f"""Fix this malformed JSON output.

ERROR: {error.message}
ORIGINAL OUTPUT: {raw_output}
TARGET SCHEMA: {schema_text}

Return ONLY valid JSON that matches the schema. No explanation."""

        response = llm_callable(prompt)

        # Try to parse the response
        extracted = self._extract_from_markdown(response)
        if extracted:
            return extracted

        return json.loads(response.strip())


class SchemaRepairStrategy(RepairStrategy):
    """Strategy for repairing schema validation errors."""

    name = "schema_repair"
    description = "Repairs outputs that don't match the expected schema"

    def can_handle(self, error: ClassifiedError) -> bool:
        return error.category in {
            ErrorCategory.SCHEMA_VALIDATION,
            ErrorCategory.MISSING_FIELD,
            ErrorCategory.TYPE_MISMATCH,
            ErrorCategory.INVALID_VALUE,
        }

    def repair(
        self,
        error: ClassifiedError,
        input_state: Any,
        raw_output: Any,
        schema: Optional[Type[BaseModel]] = None,
        llm_callable: Optional[Callable[[str], str]] = None
    ) -> Any:
        """Attempt to repair schema mismatches."""
        if not schema:
            raise ValueError("Schema repair requires a schema")

        # Try type coercion first
        if isinstance(raw_output, dict):
            coerced = self._try_coercion(raw_output, schema)
            if coerced:
                return coerced

        # Fall back to LLM
        if llm_callable:
            return self._llm_repair(error, input_state, raw_output, schema, llm_callable)

        raise ValueError(f"Could not repair schema mismatch: {error.message}")

    def _try_coercion(self, data: Dict, schema: Type[BaseModel]) -> Optional[BaseModel]:
        """Try to coerce data to match schema."""
        try:
            # Get schema fields
            fields = schema.model_fields

            coerced = {}
            for field_name, field_info in fields.items():
                if field_name in data:
                    value = data[field_name]
                    # Attempt basic type coercion
                    field_type = field_info.annotation
                    coerced[field_name] = self._coerce_value(value, field_type)
                elif field_info.default is not None:
                    coerced[field_name] = field_info.default

            return schema.model_validate(coerced)
        except Exception:
            return None

    def _coerce_value(self, value: Any, target_type: type) -> Any:
        """Attempt to coerce a value to target type."""
        if target_type == str:
            return str(value)
        elif target_type == int:
            return int(float(value))
        elif target_type == float:
            return float(value)
        elif target_type == bool:
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)
        return value

    def _llm_repair(
        self,
        error: ClassifiedError,
        input_state: Any,
        raw_output: Any,
        schema: Type[BaseModel],
        llm_callable: Callable[[str], str]
    ) -> Dict:
        """Use LLM to repair schema."""
        try:
            schema_text = json.dumps(schema.model_json_schema(), indent=2)
        except AttributeError:
            schema_text = str(schema)

        hint = ErrorClassifier.get_recovery_prompt_hint(error)

        prompt = f"""Fix this output to match the required schema.

ERROR: {error.message}
HINT: {hint}
INPUT: {json.dumps(input_state, default=str)}
FAILED OUTPUT: {json.dumps(raw_output, default=str)}
REQUIRED SCHEMA: {schema_text}

Return ONLY valid JSON matching the schema exactly. No explanation."""

        response = llm_callable(prompt)

        # Extract and parse JSON
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        return json.loads(response.strip())


class RetryWithBackoffStrategy(RepairStrategy):
    """Strategy for retrying with exponential backoff."""

    name = "retry_with_backoff"
    description = "Retries with exponential backoff for transient errors"

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self._attempt = 0

    def can_handle(self, error: ClassifiedError) -> bool:
        return error.category in {
            ErrorCategory.NETWORK_TIMEOUT,
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.API_ERROR,
            ErrorCategory.MODEL_OVERLOAD,
        }

    def repair(
        self,
        error: ClassifiedError,
        input_state: Any,
        raw_output: Any,
        schema: Optional[Type[BaseModel]] = None,
        llm_callable: Optional[Callable[[str], str]] = None
    ) -> Any:
        """Retry with backoff - mainly handles the delay logic."""
        self._attempt += 1

        if self._attempt > self.config.max_attempts:
            raise RuntimeError(f"Max retry attempts ({self.config.max_attempts}) exceeded")

        delay = self.config.get_delay(self._attempt)
        if delay > 0:
            time.sleep(delay)

        # The actual retry will be handled by the caller
        # This strategy mainly manages timing
        raise error.original_error


class TruncateContextStrategy(RepairStrategy):
    """Strategy for handling context length errors."""

    name = "truncate_context"
    description = "Truncates context to fit within limits"

    def __init__(self, max_chars: int = 8000):
        self.max_chars = max_chars

    def can_handle(self, error: ClassifiedError) -> bool:
        return error.category == ErrorCategory.CONTEXT_LENGTH

    def repair(
        self,
        error: ClassifiedError,
        input_state: Any,
        raw_output: Any,
        schema: Optional[Type[BaseModel]] = None,
        llm_callable: Optional[Callable[[str], str]] = None
    ) -> Any:
        """Truncate input state to reduce context."""
        if not isinstance(input_state, dict):
            raise ValueError("Cannot truncate non-dict input state")

        truncated = {}
        current_len = 0

        for key, value in input_state.items():
            value_str = json.dumps(value, default=str)
            if current_len + len(value_str) > self.max_chars:
                # Truncate this field
                if isinstance(value, str):
                    remaining = self.max_chars - current_len - 100
                    truncated[key] = value[:max(0, remaining)] + "...[truncated]"
                elif isinstance(value, list):
                    truncated[key] = value[:5]  # Keep first 5 items
                else:
                    truncated[key] = str(value)[:100]
                break
            else:
                truncated[key] = value
                current_len += len(value_str)

        return {"truncated_input": truncated, "requires_rerun": True}


class FailFastStrategy(RepairStrategy):
    """Strategy for unrecoverable errors - fails immediately."""

    name = "fail_fast"
    description = "Fails immediately for unrecoverable errors"

    def can_handle(self, error: ClassifiedError) -> bool:
        return error.category in {
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.MEMORY,
            ErrorCategory.LOOP_DETECTED,
        }

    def repair(
        self,
        error: ClassifiedError,
        input_state: Any,
        raw_output: Any,
        schema: Optional[Type[BaseModel]] = None,
        llm_callable: Optional[Callable[[str], str]] = None
    ) -> Any:
        """Always raise - these errors are not recoverable."""
        raise error.original_error


class LLMRepairStrategy(RepairStrategy):
    """Generic LLM-based repair strategy."""

    name = "llm_repair"
    description = "Uses LLM to repair generic errors"

    def __init__(self, custom_prompt: Optional[str] = None):
        self.custom_prompt = custom_prompt

    def can_handle(self, error: ClassifiedError) -> bool:
        # Can handle anything that's marked recoverable
        return error.recoverable

    def repair(
        self,
        error: ClassifiedError,
        input_state: Any,
        raw_output: Any,
        schema: Optional[Type[BaseModel]] = None,
        llm_callable: Optional[Callable[[str], str]] = None
    ) -> Any:
        """Use LLM for generic repair."""
        if not llm_callable:
            raise ValueError("LLM repair requires an LLM callable")

        schema_text = ""
        if schema:
            try:
                schema_text = json.dumps(schema.model_json_schema(), indent=2)
            except AttributeError:
                schema_text = str(schema)

        hint = ErrorClassifier.get_recovery_prompt_hint(error)

        if self.custom_prompt:
            prompt = self.custom_prompt.format(
                error=error.message,
                input=json.dumps(input_state, default=str),
                output=json.dumps(raw_output, default=str),
                schema=schema_text,
                hint=hint
            )
        else:
            prompt = f"""You are the AgentCircuit Medic. Fix this failed agent output.

ERROR TYPE: {error.category.value}
ERROR MESSAGE: {error.message}
HINT: {hint}

INPUT DATA: {json.dumps(input_state, default=str)}
FAILED OUTPUT: {json.dumps(raw_output, default=str)}
TARGET SCHEMA: {schema_text}

Analyze the error and return ONLY the corrected JSON output. No explanation."""

        response = llm_callable(prompt)

        # Extract JSON from response
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        return json.loads(response.strip())


@dataclass
class StrategyChain:
    """
    A chain of strategies to try in order.

    Tries each strategy until one succeeds.
    """
    strategies: List[RepairStrategy] = field(default_factory=list)
    retry_config: RetryConfig = field(default_factory=RetryConfig)

    def add(self, strategy: RepairStrategy) -> "StrategyChain":
        """Add a strategy to the chain."""
        self.strategies.append(strategy)
        return self

    def execute(
        self,
        error: ClassifiedError,
        input_state: Any,
        raw_output: Any,
        schema: Optional[Type[BaseModel]] = None,
        llm_callable: Optional[Callable[[str], str]] = None
    ) -> Any:
        """
        Execute strategies in order until one succeeds.

        Args:
            error: The classified error
            input_state: Original input
            raw_output: Failed output
            schema: Target schema
            llm_callable: LLM for repair

        Returns:
            Repaired output

        Raises:
            Exception if all strategies fail
        """
        last_error = error.original_error

        for strategy in self.strategies:
            if not strategy.can_handle(error):
                continue

            try:
                return strategy.repair(
                    error=error,
                    input_state=input_state,
                    raw_output=raw_output,
                    schema=schema,
                    llm_callable=llm_callable
                )
            except Exception as e:
                last_error = e
                continue

        raise last_error


# Default strategy chain
def create_default_chain() -> StrategyChain:
    """Create the default strategy chain."""
    return StrategyChain(
        strategies=[
            JSONRepairStrategy(),
            SchemaRepairStrategy(),
            RetryWithBackoffStrategy(),
            TruncateContextStrategy(),
            FailFastStrategy(),
            LLMRepairStrategy(),  # Fallback
        ]
    )
