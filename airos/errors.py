"""
Error categorization and classification for AirOS.

This module provides structured error types and automatic classification
to enable intelligent recovery strategies.
"""
from enum import Enum
from typing import Any, Optional, Dict, Type
from dataclasses import dataclass, field
import re
import traceback


class ErrorCategory(Enum):
    """Categories of errors for intelligent recovery."""

    # Parsing & Format Errors
    JSON_PARSE = "json_parse"
    SCHEMA_VALIDATION = "schema_validation"
    TYPE_MISMATCH = "type_mismatch"

    # Network & API Errors
    NETWORK_TIMEOUT = "network_timeout"
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    AUTHENTICATION = "authentication"

    # Execution Errors
    RUNTIME = "runtime"
    MEMORY = "memory"
    LOOP_DETECTED = "loop_detected"

    # LLM-Specific Errors
    CONTEXT_LENGTH = "context_length"
    CONTENT_FILTER = "content_filter"
    MODEL_OVERLOAD = "model_overload"

    # Data Errors
    MISSING_FIELD = "missing_field"
    INVALID_VALUE = "invalid_value"
    NULL_REFERENCE = "null_reference"

    # Unknown
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"           # Recoverable with simple retry
    MEDIUM = "medium"     # Recoverable with intervention
    HIGH = "high"         # Requires significant intervention
    CRITICAL = "critical" # Likely unrecoverable


@dataclass
class ClassifiedError:
    """A classified error with metadata for recovery."""

    original_error: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    recoverable: bool = True
    retry_suggested: bool = True
    suggested_strategy: str = "llm_repair"
    context: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.category.value}:{self.severity.value}] {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "recoverable": self.recoverable,
            "retry_suggested": self.retry_suggested,
            "suggested_strategy": self.suggested_strategy,
            "original_type": type(self.original_error).__name__,
            "context": self.context
        }


class ErrorClassifier:
    """
    Classifies errors into categories for intelligent recovery.

    Uses pattern matching on error messages and exception types
    to determine the best recovery strategy.
    """

    # Pattern matchers for error classification
    PATTERNS: Dict[ErrorCategory, list] = {
        ErrorCategory.JSON_PARSE: [
            r"json\.decoder\.JSONDecodeError",
            r"Expecting.*JSON",
            r"Invalid JSON",
            r"json\.loads",
            r"parse.*json",
            r"JSONDecodeError",
        ],
        ErrorCategory.SCHEMA_VALIDATION: [
            r"ValidationError",
            r"Sentinel Alert",
            r"validation failed",
            r"pydantic",
            r"field required",
            r"missing.*field",
        ],
        ErrorCategory.TYPE_MISMATCH: [
            r"type.*expected",
            r"TypeError",
            r"cannot convert",
            r"invalid type",
            r"wrong type",
        ],
        ErrorCategory.NETWORK_TIMEOUT: [
            r"timeout",
            r"timed out",
            r"connection.*reset",
            r"ReadTimeout",
            r"ConnectTimeout",
        ],
        ErrorCategory.RATE_LIMIT: [
            r"rate.*limit",
            r"429",
            r"too many requests",
            r"quota.*exceeded",
            r"RateLimitError",
        ],
        ErrorCategory.API_ERROR: [
            r"API.*error",
            r"500",
            r"502",
            r"503",
            r"BadGateway",
            r"ServiceUnavailable",
        ],
        ErrorCategory.AUTHENTICATION: [
            r"401",
            r"403",
            r"unauthorized",
            r"forbidden",
            r"AuthenticationError",
            r"invalid.*key",
            r"api.*key",
        ],
        ErrorCategory.CONTEXT_LENGTH: [
            r"context.*length",
            r"maximum.*token",
            r"too.*long",
            r"context_length_exceeded",
            r"max_tokens",
        ],
        ErrorCategory.CONTENT_FILTER: [
            r"content.*filter",
            r"safety.*filter",
            r"blocked",
            r"inappropriate",
            r"ContentFilterError",
        ],
        ErrorCategory.MODEL_OVERLOAD: [
            r"overloaded",
            r"capacity",
            r"try again",
            r"model.*busy",
        ],
        ErrorCategory.MEMORY: [
            r"MemoryError",
            r"out of memory",
            r"OOM",
            r"memory.*exceeded",
        ],
        ErrorCategory.LOOP_DETECTED: [
            r"loop.*detected",
            r"Fuse.*Tripped",
            r"infinite.*loop",
            r"cycle.*detected",
        ],
        ErrorCategory.MISSING_FIELD: [
            r"KeyError",
            r"missing.*key",
            r"required.*field",
            r"'.*' not found",
        ],
        ErrorCategory.NULL_REFERENCE: [
            r"NoneType",
            r"null.*reference",
            r"AttributeError.*None",
        ],
    }

    # Severity mapping for categories
    SEVERITY_MAP: Dict[ErrorCategory, ErrorSeverity] = {
        ErrorCategory.JSON_PARSE: ErrorSeverity.LOW,
        ErrorCategory.SCHEMA_VALIDATION: ErrorSeverity.LOW,
        ErrorCategory.TYPE_MISMATCH: ErrorSeverity.MEDIUM,
        ErrorCategory.NETWORK_TIMEOUT: ErrorSeverity.MEDIUM,
        ErrorCategory.RATE_LIMIT: ErrorSeverity.MEDIUM,
        ErrorCategory.API_ERROR: ErrorSeverity.MEDIUM,
        ErrorCategory.AUTHENTICATION: ErrorSeverity.HIGH,
        ErrorCategory.CONTEXT_LENGTH: ErrorSeverity.MEDIUM,
        ErrorCategory.CONTENT_FILTER: ErrorSeverity.HIGH,
        ErrorCategory.MODEL_OVERLOAD: ErrorSeverity.LOW,
        ErrorCategory.MEMORY: ErrorSeverity.CRITICAL,
        ErrorCategory.LOOP_DETECTED: ErrorSeverity.HIGH,
        ErrorCategory.MISSING_FIELD: ErrorSeverity.LOW,
        ErrorCategory.NULL_REFERENCE: ErrorSeverity.MEDIUM,
        ErrorCategory.INVALID_VALUE: ErrorSeverity.LOW,
        ErrorCategory.RUNTIME: ErrorSeverity.MEDIUM,
        ErrorCategory.UNKNOWN: ErrorSeverity.MEDIUM,
    }

    # Recovery strategy suggestions
    STRATEGY_MAP: Dict[ErrorCategory, str] = {
        ErrorCategory.JSON_PARSE: "json_repair",
        ErrorCategory.SCHEMA_VALIDATION: "schema_repair",
        ErrorCategory.TYPE_MISMATCH: "type_coercion",
        ErrorCategory.NETWORK_TIMEOUT: "retry_with_backoff",
        ErrorCategory.RATE_LIMIT: "retry_with_backoff",
        ErrorCategory.API_ERROR: "retry_with_backoff",
        ErrorCategory.AUTHENTICATION: "fail_fast",
        ErrorCategory.CONTEXT_LENGTH: "truncate_context",
        ErrorCategory.CONTENT_FILTER: "sanitize_input",
        ErrorCategory.MODEL_OVERLOAD: "retry_with_backoff",
        ErrorCategory.MEMORY: "fail_fast",
        ErrorCategory.LOOP_DETECTED: "fail_fast",
        ErrorCategory.MISSING_FIELD: "llm_repair",
        ErrorCategory.NULL_REFERENCE: "llm_repair",
        ErrorCategory.INVALID_VALUE: "llm_repair",
        ErrorCategory.RUNTIME: "llm_repair",
        ErrorCategory.UNKNOWN: "llm_repair",
    }

    # Non-recoverable categories
    NON_RECOVERABLE = {
        ErrorCategory.AUTHENTICATION,
        ErrorCategory.MEMORY,
        ErrorCategory.LOOP_DETECTED,
    }

    @classmethod
    def classify(cls, error: Exception, context: Optional[Dict[str, Any]] = None) -> ClassifiedError:
        """
        Classify an error and return structured metadata.

        Args:
            error: The exception to classify
            context: Optional additional context about the error

        Returns:
            ClassifiedError with category, severity, and recovery suggestions
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        full_text = f"{error_type} {error_str}"

        # Try to match against patterns
        matched_category = ErrorCategory.UNKNOWN
        for category, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, full_text, re.IGNORECASE):
                    matched_category = category
                    break
            if matched_category != ErrorCategory.UNKNOWN:
                break

        # Get severity and strategy
        severity = cls.SEVERITY_MAP.get(matched_category, ErrorSeverity.MEDIUM)
        strategy = cls.STRATEGY_MAP.get(matched_category, "llm_repair")
        recoverable = matched_category not in cls.NON_RECOVERABLE

        # Determine if retry is suggested
        retry_suggested = recoverable and severity in {ErrorSeverity.LOW, ErrorSeverity.MEDIUM}

        return ClassifiedError(
            original_error=error,
            category=matched_category,
            severity=severity,
            message=str(error),
            recoverable=recoverable,
            retry_suggested=retry_suggested,
            suggested_strategy=strategy,
            context=context or {}
        )

    @classmethod
    def get_recovery_prompt_hint(cls, classified: ClassifiedError) -> str:
        """
        Get a hint for the LLM repair prompt based on error category.

        Args:
            classified: The classified error

        Returns:
            A string hint to include in the repair prompt
        """
        hints = {
            ErrorCategory.JSON_PARSE: "The output must be valid JSON. Check for missing quotes, brackets, or commas.",
            ErrorCategory.SCHEMA_VALIDATION: "The output must match the required schema exactly. Check field names and types.",
            ErrorCategory.TYPE_MISMATCH: "Ensure all values have the correct types (strings, numbers, booleans).",
            ErrorCategory.MISSING_FIELD: "All required fields must be present in the output.",
            ErrorCategory.NULL_REFERENCE: "Ensure no null/None values where they're not allowed.",
            ErrorCategory.INVALID_VALUE: "Check that all values are within valid ranges and formats.",
            ErrorCategory.CONTEXT_LENGTH: "The response may need to be shorter. Focus on essential information only.",
            ErrorCategory.CONTENT_FILTER: "Rephrase the content to be appropriate and avoid sensitive topics.",
        }

        return hints.get(classified.category, "Review the error and correct the output accordingly.")


# Custom error classes for AirOS
class AirOSError(Exception):
    """Base exception for AirOS errors."""
    pass


class RecoveryError(AirOSError):
    """Raised when recovery fails."""
    def __init__(self, message: str, classified_error: Optional[ClassifiedError] = None):
        super().__init__(message)
        self.classified_error = classified_error


class ConfigurationError(AirOSError):
    """Raised for configuration issues."""
    pass


class ProviderError(AirOSError):
    """Raised for LLM provider issues."""
    def __init__(self, message: str, provider: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.provider = provider
        self.original_error = original_error


class MedicError(AirOSError):
    """Raised when Medic recovery fails after all attempts."""
    pass


class BudgetExceededError(AirOSError):
    """Raised when cost budget is exceeded."""
    def __init__(self, message: str, spent: float = 0.0, limit: float = 0.0):
        super().__init__(message)
        self.spent = spent
        self.limit = limit


class TimeoutExceededError(AirOSError):
    """Raised when execution time limit is exceeded."""
    def __init__(self, message: str, elapsed: float = 0.0, limit: float = 0.0):
        super().__init__(message)
        self.elapsed = elapsed
        self.limit = limit
