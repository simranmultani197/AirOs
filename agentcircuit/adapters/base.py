"""
Base adapter interface for framework integrations.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel


@dataclass
class AdapterConfig:
    """Configuration for framework adapters."""

    # Fuse settings
    fuse_enabled: bool = True
    fuse_limit: int = 3

    # Medic settings
    medic_enabled: bool = True
    llm_callable: Optional[Callable[[str], str]] = None
    model: Optional[str] = None
    fallback_models: Optional[List[str]] = None
    max_recovery_attempts: int = 2

    # Sentinel settings
    sentinel_enabled: bool = True
    default_schema: Optional[Type[BaseModel]] = None

    # Storage settings
    storage_enabled: bool = True
    db_path: Optional[str] = None

    # Tracking
    track_costs: bool = True

    # Extra settings
    extra: Dict[str, Any] = field(default_factory=dict)


T = TypeVar("T")


class FrameworkAdapter(ABC):
    """
    Abstract base class for framework adapters.

    Each framework adapter provides:
    - A way to wrap/decorate nodes or agents
    - A middleware/plugin interface if supported
    - Automatic state extraction
    - Framework-specific error handling
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        self.config = config or AdapterConfig()

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Return the name of the framework."""
        pass

    @abstractmethod
    def wrap_node(
        self,
        func: Callable,
        schema: Optional[Type[BaseModel]] = None,
        node_name: Optional[str] = None,
        **kwargs
    ) -> Callable:
        """
        Wrap a single node/function with AgentCircuit reliability.

        Args:
            func: The function to wrap
            schema: Optional output schema
            node_name: Optional custom name
            **kwargs: Additional options

        Returns:
            Wrapped function
        """
        pass

    @abstractmethod
    def create_middleware(self) -> Any:
        """
        Create a middleware component for the framework.

        Returns framework-specific middleware that can be attached
        to the entire graph/agent.
        """
        pass

    @abstractmethod
    def extract_state(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Extract state from framework-specific arguments.

        Args:
            *args: Positional arguments passed to the node
            **kwargs: Keyword arguments passed to the node

        Returns:
            Normalized state dictionary
        """
        pass

    @abstractmethod
    def extract_config(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Extract configuration from framework-specific arguments.

        Args:
            *args: Positional arguments passed to the node
            **kwargs: Keyword arguments passed to the node

        Returns:
            Configuration dictionary with run_id, thread_id, etc.
        """
        pass

    def get_run_id(self, config: Dict[str, Any]) -> str:
        """
        Get run ID from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Run identifier string
        """
        return (
            config.get("configurable", {}).get("thread_id")
            or config.get("run_id")
            or config.get("session_id")
            or "default_run"
        )


class AdapterRegistry:
    """Registry of available framework adapters."""

    _adapters: Dict[str, Type[FrameworkAdapter]] = {}

    @classmethod
    def register(cls, name: str, adapter_class: Type[FrameworkAdapter]) -> None:
        """Register an adapter."""
        cls._adapters[name.lower()] = adapter_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[FrameworkAdapter]]:
        """Get an adapter by name."""
        return cls._adapters.get(name.lower())

    @classmethod
    def list_adapters(cls) -> List[str]:
        """List available adapters."""
        return list(cls._adapters.keys())

    @classmethod
    def create(cls, name: str, config: Optional[AdapterConfig] = None) -> FrameworkAdapter:
        """Create an adapter instance."""
        adapter_class = cls.get(name)
        if not adapter_class:
            raise ValueError(f"Unknown adapter: {name}. Available: {cls.list_adapters()}")
        return adapter_class(config)
