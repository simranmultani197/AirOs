"""
AirOS - One decorator to make any AI agent reliable.

Usage:
    from airos import reliable

    @reliable(fuse_limit=3)
    def my_agent_node(state):
        return {"result": "done"}

Or with schema validation:

    from airos import reliable
    from pydantic import BaseModel

    class Output(BaseModel):
        answer: str
        confidence: float

    @reliable(sentinel_schema=Output)
    def my_agent_node(state):
        return {"answer": "hello", "confidence": 0.95}
"""

# Core - always available, zero heavy deps
from .core import reliable, reliable_node
from .fuse import Fuse, LoopError
from .sentinel import Sentinel, SentinelError
from .storage import (
    InMemoryStorage,
    Storage,
    BaseStorage,
    create_storage,
    get_default_storage,
    set_default_storage,
)

# Medic - core component (depends only on pydantic)
from .medic import Medic, MedicError, RecoveryResult

# Budget - cost-saving circuit breakers
from .budget import BudgetFuse, TimeoutFuse, GlobalBudget

# Pricing - model cost calculation
from .pricing import CostCalculator, ModelPricing, MODEL_PRICING, get_model_pricing

# Error handling - lightweight, no external deps
from .errors import (
    ErrorCategory,
    ErrorSeverity,
    ClassifiedError,
    ErrorClassifier,
    AirOSError,
    RecoveryError,
    ConfigurationError,
    ProviderError,
    BudgetExceededError,
    TimeoutExceededError,
)

# Strategies - lightweight
from .strategies import (
    RepairStrategy,
    StrategyChain,
    RetryConfig,
    RetryStrategy,
    JSONRepairStrategy,
    SchemaRepairStrategy,
    LLMRepairStrategy,
    RetryWithBackoffStrategy,
    TruncateContextStrategy,
    FailFastStrategy,
    create_default_chain as create_default_strategy_chain,
)


# --- Lazy imports for heavy/optional modules ---
# These are only loaded when accessed, so `pip install airos` stays fast

def __getattr__(name):
    """Lazy-load optional modules on first access."""

    # Providers (require openai/anthropic/groq SDKs)
    _provider_names = {
        "LLMProvider", "ProviderChain", "ProviderType", "ModelConfig",
        "TokenUsage", "create_provider", "create_default_provider_chain",
        "get_model", "MODELS",
    }
    if name in _provider_names:
        from . import providers
        return getattr(providers, name)

    # RCA (lightweight but not always needed)
    _rca_names = {
        "RootCauseAnalyzer", "RCAReport", "RootCause", "RCACategory",
        "analyze_failures",
    }
    if name in _rca_names:
        from . import rca
        return getattr(rca, name)

    raise AttributeError(f"module 'airos' has no attribute {name!r}")


# Version
__version__ = "0.4.1"

__all__ = [
    # Primary API
    "reliable",
    "reliable_node",
    # Core components
    "Fuse",
    "LoopError",
    "Medic",
    "MedicError",
    "RecoveryResult",
    "Sentinel",
    "SentinelError",
    # Budget / Cost-saving
    "BudgetFuse",
    "TimeoutFuse",
    "GlobalBudget",
    # Pricing
    "CostCalculator",
    "ModelPricing",
    "MODEL_PRICING",
    "get_model_pricing",
    # Storage
    "InMemoryStorage",
    "Storage",
    "BaseStorage",
    "create_storage",
    "get_default_storage",
    "set_default_storage",
    # Errors
    "ErrorCategory",
    "ErrorSeverity",
    "ClassifiedError",
    "ErrorClassifier",
    "AirOSError",
    "RecoveryError",
    "ConfigurationError",
    "ProviderError",
    "BudgetExceededError",
    "TimeoutExceededError",
    # Strategies
    "RepairStrategy",
    "StrategyChain",
    "RetryConfig",
    "RetryStrategy",
    "JSONRepairStrategy",
    "SchemaRepairStrategy",
    "LLMRepairStrategy",
    "RetryWithBackoffStrategy",
    "TruncateContextStrategy",
    "FailFastStrategy",
    "create_default_strategy_chain",
    # Version
    "__version__",
]


def get_adapter(framework: str, **kwargs):
    """
    Get a framework adapter.

    Args:
        framework: Framework name (langgraph, langchain, crewai, autogen)
        **kwargs: Adapter configuration options

    Returns:
        Configured FrameworkAdapter instance

    Example:
        adapter = get_adapter("langchain", fuse_limit=5)
        wrapped_chain = adapter.wrap_chain(my_chain)
    """
    from .adapters import AdapterRegistry, AdapterConfig

    config = AdapterConfig(**kwargs) if kwargs else None
    return AdapterRegistry.create(framework, config)
