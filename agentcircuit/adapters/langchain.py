"""
LangChain adapter for AgentCircuit.

Provides integration with LangChain's LCEL chains, agents, and tools.
"""
import functools
from typing import Any, Callable, Dict, List, Optional, Type, Union
import json
import time
import traceback

from pydantic import BaseModel

from .base import FrameworkAdapter, AdapterConfig, AdapterRegistry
from ..fuse import Fuse, LoopError
from ..medic import Medic, MedicError
from ..sentinel import Sentinel, SentinelError
from ..storage import Storage


class LangChainAdapter(FrameworkAdapter):
    """
    Adapter for LangChain framework.

    Supports:
    - LCEL Runnables
    - Tools
    - Agents
    - Chains
    """

    @property
    def framework_name(self) -> str:
        return "langchain"

    def wrap_node(
        self,
        func: Callable,
        schema: Optional[Type[BaseModel]] = None,
        node_name: Optional[str] = None,
        **kwargs
    ) -> Callable:
        """Wrap a LangChain component with reliability features."""
        return self._create_wrapper(func, schema, node_name)

    def create_middleware(self) -> "LangChainMiddleware":
        """Create middleware for LangChain."""
        return LangChainMiddleware(self.config)

    def extract_state(self, *args, **kwargs) -> Dict[str, Any]:
        """Extract state from LangChain invocation."""
        # LangChain typically uses input as first arg or "input" kwarg
        if args:
            first_arg = args[0]
            if isinstance(first_arg, dict):
                return first_arg
            elif isinstance(first_arg, str):
                return {"input": first_arg}
            elif hasattr(first_arg, "dict"):
                return first_arg.dict()

        return kwargs.get("input", kwargs)

    def extract_config(self, *args, **kwargs) -> Dict[str, Any]:
        """Extract config from LangChain invocation."""
        config = kwargs.get("config", {})

        # LangChain uses RunnableConfig
        if hasattr(config, "get"):
            return {
                "run_id": config.get("run_id"),
                "configurable": config.get("configurable", {}),
                "tags": config.get("tags", []),
                "metadata": config.get("metadata", {})
            }

        return {"configurable": {}}

    def _create_wrapper(
        self,
        func: Callable,
        schema: Optional[Type[BaseModel]],
        node_name: Optional[str]
    ) -> Callable:
        """Create a wrapper function for LangChain components."""
        actual_name = node_name or getattr(func, "__name__", "langchain_node")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize components
            storage = Storage(db_path=self.config.db_path) if self.config.storage_enabled else None
            fuse = Fuse(limit=self.config.fuse_limit) if self.config.fuse_enabled else None
            medic = Medic(
                llm_callable=self.config.llm_callable,
                model=self.config.model,
                fallback_models=self.config.fallback_models
            ) if self.config.medic_enabled else None
            sentinel = Sentinel(schema=schema) if self.config.sentinel_enabled else None

            # Extract state and config
            state = self.extract_state(*args, **kwargs)
            config = self.extract_config(*args, **kwargs)
            run_id = self.get_run_id(config)

            start_time = time.time()
            result = None
            status = "success"
            recovery_count = 0
            diagnosis = None

            # Fuse check
            if fuse and storage:
                history = storage.get_run_history(run_id)
                node_history = [
                    fuse._hash_state(h["input_state"])
                    for h in history if h["node_id"] == actual_name
                ]
                try:
                    fuse.check(history=node_history, current_state=state)
                except LoopError as e:
                    if storage:
                        storage.log_trace(
                            run_id=run_id,
                            node_id=actual_name,
                            input_state=state,
                            output_state=None,
                            status="failed_loop",
                            recovery_attempts=0
                        )
                    raise e

            # Execute with recovery
            current_error = None
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                current_error = e
                diagnosis = str(e)

            # Validate if no error
            if not current_error and sentinel:
                try:
                    result = sentinel.validate(result)
                except SentinelError as e:
                    current_error = e
                    diagnosis = str(e)

            # Recovery loop
            while current_error and medic and recovery_count < self.config.max_recovery_attempts:
                recovery_count += 1
                try:
                    raw_output = result if isinstance(current_error, SentinelError) else None
                    fixed = medic.attempt_recovery(
                        error=current_error,
                        input_state=state,
                        raw_output=raw_output,
                        node_id=actual_name,
                        recovery_attempts=recovery_count,
                        schema=schema
                    )
                    if sentinel:
                        result = sentinel.validate(fixed)
                    else:
                        result = fixed
                    current_error = None
                    status = "repaired"
                except Exception as e:
                    current_error = e
                    diagnosis = str(e)

            # Log and return
            duration_ms = (time.time() - start_time) * 1000

            if current_error:
                if storage:
                    storage.log_trace(
                        run_id=run_id,
                        node_id=actual_name,
                        input_state=state,
                        output_state=str(current_error),
                        status="failed",
                        recovery_attempts=recovery_count,
                        diagnosis=diagnosis,
                        duration_ms=duration_ms
                    )
                raise current_error

            if storage:
                storage.log_trace(
                    run_id=run_id,
                    node_id=actual_name,
                    input_state=state,
                    output_state=result if isinstance(result, dict) else {"output": str(result)},
                    status=status,
                    recovery_attempts=recovery_count,
                    diagnosis=diagnosis,
                    duration_ms=duration_ms
                )

            return result

        return wrapper

    def wrap_tool(self, tool: Any, schema: Optional[Type[BaseModel]] = None) -> Any:
        """
        Wrap a LangChain Tool with reliability features.

        Args:
            tool: LangChain Tool or BaseTool instance
            schema: Optional output schema

        Returns:
            Wrapped tool
        """
        try:
            from langchain_core.tools import BaseTool
        except ImportError:
            raise ImportError("langchain-core is required. Run: pip install langchain-core")

        if hasattr(tool, "_run"):
            original_run = tool._run
            tool._run = self._create_wrapper(original_run, schema, tool.name)

        return tool

    def wrap_chain(self, chain: Any, schema: Optional[Type[BaseModel]] = None) -> Any:
        """
        Wrap a LangChain chain/runnable with reliability features.

        Args:
            chain: LangChain Runnable
            schema: Optional output schema

        Returns:
            Wrapped runnable
        """
        try:
            from langchain_core.runnables import RunnableLambda
        except ImportError:
            raise ImportError("langchain-core is required. Run: pip install langchain-core")

        wrapped_invoke = self._create_wrapper(
            chain.invoke,
            schema,
            getattr(chain, "name", "langchain_chain")
        )

        # Create new runnable with wrapped invoke
        return RunnableLambda(wrapped_invoke)


class LangChainMiddleware:
    """
    Middleware for automatically wrapping LangChain components.

    Usage:
        from agentcircuit.adapters import LangChainAdapter

        adapter = LangChainAdapter()
        middleware = adapter.create_middleware()

        # Wrap a chain
        chain = middleware.wrap_chain(my_chain)

        # Or use as decorator
        @middleware.tool(schema=MySchema)
        def my_tool(input: str) -> str:
            return "result"
    """

    def __init__(self, config: AdapterConfig):
        self.config = config
        self._adapter = LangChainAdapter(config)

    def wrap_chain(self, chain: Any, schema: Optional[Type[BaseModel]] = None) -> Any:
        """Wrap a LangChain chain."""
        return self._adapter.wrap_chain(chain, schema)

    def wrap_tool(self, tool: Any, schema: Optional[Type[BaseModel]] = None) -> Any:
        """Wrap a LangChain tool."""
        return self._adapter.wrap_tool(tool, schema)

    def tool(
        self,
        schema: Optional[Type[BaseModel]] = None,
        name: Optional[str] = None
    ) -> Callable:
        """Decorator for wrapping tool functions."""
        def decorator(func: Callable) -> Callable:
            return self._adapter.wrap_node(func, schema, name or func.__name__)
        return decorator

    def chain(
        self,
        schema: Optional[Type[BaseModel]] = None,
        name: Optional[str] = None
    ) -> Callable:
        """Decorator for wrapping chain functions."""
        return self.tool(schema, name)


# Register the adapter
AdapterRegistry.register("langchain", LangChainAdapter)
