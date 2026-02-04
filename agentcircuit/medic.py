"""
The Medic: Advanced LLM-based error recovery for AgentCircuit.

This module provides intelligent error recovery using:
- Error classification and categorization
- Multiple repair strategies
- Multi-model LLM support with fallback
- Real cost tracking
"""
from typing import Callable, Any, Optional, Dict, List, Type, Union
from dataclasses import dataclass, field
import json
import os
import time

from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .errors import (
    ErrorClassifier,
    ClassifiedError,
    ErrorCategory,
    ErrorSeverity,
    MedicError,
    RecoveryError,
    ProviderError
)
from .strategies import (
    RepairStrategy,
    StrategyChain,
    RetryConfig,
    JSONRepairStrategy,
    SchemaRepairStrategy,
    LLMRepairStrategy,
    RetryWithBackoffStrategy,
    create_default_chain as create_default_strategy_chain
)

# Import providers lazily to avoid circular imports
_providers_module = None


def _get_providers():
    global _providers_module
    if _providers_module is None:
        from . import providers as _providers_module
    return _providers_module


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    success: bool
    output: Any = None
    attempts: int = 0
    strategy_used: Optional[str] = None
    total_time_ms: float = 0.0
    token_usage: int = 0
    estimated_cost: float = 0.0
    error_category: Optional[str] = None
    diagnosis: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "attempts": self.attempts,
            "strategy_used": self.strategy_used,
            "total_time_ms": self.total_time_ms,
            "token_usage": self.token_usage,
            "estimated_cost": self.estimated_cost,
            "error_category": self.error_category,
            "diagnosis": self.diagnosis
        }


class Medic:
    """
    The Medic: Attempts to repair errors using intelligent strategies.

    Features:
    - Automatic error classification
    - Strategy-based recovery
    - Multi-model LLM support with fallback
    - Real cost tracking
    """

    def __init__(
        self,
        llm_callable: Optional[Callable[[str], str]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        strategies: Optional[List[RepairStrategy]] = None,
        retry_config: Optional[RetryConfig] = None,
        max_recovery_attempts: int = 2,
        track_costs: bool = True
    ):
        """
        Initialize the Medic.

        Args:
            llm_callable: Custom LLM callable (takes prompt, returns response)
            model: Model shortcut name (e.g., "gpt-4o", "claude-3-5-sonnet")
            provider: Provider type if not using shortcut (openai, anthropic, groq, ollama)
            fallback_models: List of fallback model names
            strategies: Custom repair strategies (uses default if None)
            retry_config: Retry configuration
            max_recovery_attempts: Maximum recovery attempts (default 2)
            track_costs: Whether to track token costs
        """
        self.max_recovery_attempts = max_recovery_attempts
        self.track_costs = track_costs
        self.retry_config = retry_config or RetryConfig()

        # Set up LLM callable
        self.llm_callable = self._setup_llm(llm_callable, model, provider, fallback_models)

        # Set up strategies
        self.strategy_chain = self._setup_strategies(strategies)

        # Tracking
        self._total_tokens = 0
        self._total_cost = 0.0
        self._recovery_history: List[RecoveryResult] = []

    def _setup_llm(
        self,
        llm_callable: Optional[Callable[[str], str]],
        model: Optional[str],
        provider: Optional[str],
        fallback_models: Optional[List[str]]
    ) -> Optional[Callable[[str], str]]:
        """Set up the LLM callable with optional fallbacks."""
        # If custom callable provided, use it
        if llm_callable:
            return llm_callable

        providers = _get_providers()

        # Try to create a provider chain
        try:
            if model:
                # Use named model
                chain = providers.ProviderChain()
                chain.add(providers.get_model(model))

                # Add fallbacks
                if fallback_models:
                    for fb_model in fallback_models:
                        try:
                            chain.add(providers.get_model(fb_model))
                        except Exception:
                            pass

                return chain

            elif provider:
                # Create specific provider
                return providers.create_provider(provider, model or "default")

            else:
                # Try to create default chain
                try:
                    return providers.create_default_chain()
                except ProviderError:
                    pass

        except Exception as e:
            print(f"Medic: Could not initialize LLM provider: {e}")

        # Legacy fallback to Groq
        try:
            from groq import Groq
            if os.environ.get("GROQ_API_KEY"):
                print("Medic: Falling back to Groq (llama-3.3-70b-versatile).")
                client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                return lambda prompt: client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile"
                ).choices[0].message.content
        except ImportError:
            pass

        return None

    def _setup_strategies(
        self,
        strategies: Optional[List[RepairStrategy]]
    ) -> StrategyChain:
        """Set up the strategy chain."""
        if strategies:
            return StrategyChain(strategies=strategies, retry_config=self.retry_config)
        return create_default_strategy_chain()

    def attempt_recovery(
        self,
        error: Exception,
        input_state: Any,
        raw_output: Any,
        node_id: str,
        recovery_attempts: int,
        schema: Optional[Type[BaseModel]] = None
    ) -> Dict[str, Any]:
        """
        Attempt to repair a failed node execution.

        Args:
            error: The exception that occurred
            input_state: Original input state
            raw_output: The failed output (if any)
            node_id: Name of the failed node
            recovery_attempts: Current attempt number
            schema: Target Pydantic schema

        Returns:
            Repaired output as dictionary

        Raises:
            MedicError if recovery fails
        """
        start_time = time.time()

        # Check limits
        if recovery_attempts > self.max_recovery_attempts:
            raise MedicError(
                f"Medic: Critical Failure. Exceeded {self.max_recovery_attempts} recovery attempts. "
                f"Original error: {error}"
            )

        if not self.llm_callable:
            raise error

        print(f"Medic: Initiating Repair Sequence (Attempt {recovery_attempts})...")

        # Classify the error
        classified = ErrorClassifier.classify(
            error,
            context={"node_id": node_id, "attempt": recovery_attempts}
        )

        print(f"Medic: Error classified as [{classified.category.value}] - {classified.severity.value}")

        # Check if error is recoverable
        if not classified.recoverable:
            print(f"Medic: Error is not recoverable. Category: {classified.category.value}")
            raise error

        # Try strategy chain
        try:
            result = self.strategy_chain.execute(
                error=classified,
                input_state=input_state,
                raw_output=raw_output,
                schema=schema,
                llm_callable=self.llm_callable
            )

            # Track results
            elapsed_ms = (time.time() - start_time) * 1000

            recovery_result = RecoveryResult(
                success=True,
                output=result,
                attempts=recovery_attempts,
                strategy_used=classified.suggested_strategy,
                total_time_ms=elapsed_ms,
                error_category=classified.category.value,
                diagnosis=classified.message
            )

            self._recovery_history.append(recovery_result)

            return result

        except Exception as e:
            # Strategy chain failed, try direct LLM repair
            return self._direct_llm_repair(
                classified=classified,
                input_state=input_state,
                raw_output=raw_output,
                node_id=node_id,
                schema=schema,
                start_time=start_time
            )

    def _direct_llm_repair(
        self,
        classified: ClassifiedError,
        input_state: Any,
        raw_output: Any,
        node_id: str,
        schema: Optional[Type[BaseModel]],
        start_time: float
    ) -> Dict[str, Any]:
        """Direct LLM-based repair as fallback."""
        # Build schema text
        schema_text = ""
        if schema:
            try:
                schema_text = json.dumps(schema.model_json_schema(), indent=2)
            except AttributeError:
                try:
                    schema_text = schema.schema_json(indent=2)
                except Exception:
                    schema_text = str(schema)

        # Get hint for this error type
        hint = ErrorClassifier.get_recovery_prompt_hint(classified)

        prompt = f"""SYSTEM: You are the AgentCircuit Medic. Your job is to fix a failed agent node.

ERROR CATEGORY: {classified.category.value}
ERROR SEVERITY: {classified.severity.value}
ERROR MESSAGE: {classified.message}
HINT: {hint}

CONTEXT:
- Node: '{node_id}'
- Input: {json.dumps(input_state, default=str)[:2000]}
- Previous Output: {json.dumps(raw_output, default=str)[:2000]}

TARGET SCHEMA:
{schema_text}

INSTRUCTION: Analyze the error and fix the output. Return ONLY valid JSON matching the schema. No explanation or markdown."""

        try:
            repair_str = self.llm_callable(prompt)

            # Parse response
            result = self._parse_llm_response(repair_str)

            # Track results
            elapsed_ms = (time.time() - start_time) * 1000

            # Estimate tokens (rough)
            prompt_tokens = len(prompt) // 4
            response_tokens = len(repair_str) // 4

            recovery_result = RecoveryResult(
                success=True,
                output=result,
                attempts=1,
                strategy_used="direct_llm_repair",
                total_time_ms=elapsed_ms,
                token_usage=prompt_tokens + response_tokens,
                estimated_cost=(prompt_tokens + response_tokens) * 0.000005,
                error_category=classified.category.value,
                diagnosis=classified.message
            )

            self._recovery_history.append(recovery_result)
            self._total_tokens += recovery_result.token_usage
            self._total_cost += recovery_result.estimated_cost

            return result

        except Exception as e:
            raise MedicError(
                f"Medic: Repair failed. Error: {e}"
            ) from e

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        # Clean up markdown
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        return json.loads(response.strip())

    @property
    def total_tokens_used(self) -> int:
        """Get total tokens used across all recoveries."""
        return self._total_tokens

    @property
    def total_cost(self) -> float:
        """Get total estimated cost across all recoveries."""
        return self._total_cost

    @property
    def recovery_history(self) -> List[RecoveryResult]:
        """Get history of recovery attempts."""
        return self._recovery_history.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        successful = [r for r in self._recovery_history if r.success]
        failed = [r for r in self._recovery_history if not r.success]

        return {
            "total_attempts": len(self._recovery_history),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self._recovery_history) if self._recovery_history else 0,
            "total_tokens": self._total_tokens,
            "total_cost": self._total_cost,
            "avg_recovery_time_ms": sum(r.total_time_ms for r in self._recovery_history) / len(self._recovery_history) if self._recovery_history else 0,
            "categories": self._get_category_breakdown()
        }

    def _get_category_breakdown(self) -> Dict[str, int]:
        """Get breakdown of errors by category."""
        breakdown = {}
        for result in self._recovery_history:
            cat = result.error_category or "unknown"
            breakdown[cat] = breakdown.get(cat, 0) + 1
        return breakdown


# MedicError is now imported from errors.py for backward compatibility
