"""
LangGraph adapter for AirOS.

This is the native/primary integration as AirOS was originally designed for LangGraph.
"""
import functools
from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel

from .base import FrameworkAdapter, AdapterConfig, AdapterRegistry
from ..core import reliable_node
from ..fuse import Fuse, LoopError
from ..medic import Medic
from ..sentinel import Sentinel
from ..storage import Storage


class LangGraphAdapter(FrameworkAdapter):
    """
    Adapter for LangGraph framework.

    LangGraph is the native framework for AirOS, so this adapter
    mainly wraps the existing reliable_node decorator.
    """

    @property
    def framework_name(self) -> str:
        return "langgraph"

    def wrap_node(
        self,
        func: Callable,
        schema: Optional[Type[BaseModel]] = None,
        node_name: Optional[str] = None,
        **kwargs
    ) -> Callable:
        """Wrap a LangGraph node with reliability features."""
        return reliable_node(
            sentinel_schema=schema or self.config.default_schema,
            llm_callable=self.config.llm_callable,
            fuse_limit=self.config.fuse_limit,
            node_name=node_name
        )(func)

    def create_middleware(self) -> "LangGraphMiddleware":
        """Create middleware for automatic node wrapping."""
        return LangGraphMiddleware(self.config)

    def extract_state(self, *args, **kwargs) -> Dict[str, Any]:
        """Extract state from LangGraph node arguments."""
        if args:
            state = args[0]
            if isinstance(state, dict):
                return state
        return kwargs.get("state", {})

    def extract_config(self, *args, **kwargs) -> Dict[str, Any]:
        """Extract config from LangGraph node arguments."""
        # Check args for config
        for arg in args:
            if isinstance(arg, dict) and "configurable" in arg:
                return arg

        return kwargs.get("config", {})


class LangGraphMiddleware:
    """
    Middleware for LangGraph that automatically wraps all nodes.

    Usage:
        from langgraph.graph import StateGraph
        from airos.adapters import LangGraphAdapter

        adapter = LangGraphAdapter()
        middleware = adapter.create_middleware()

        graph = StateGraph(...)
        middleware.attach(graph)
    """

    def __init__(self, config: AdapterConfig):
        self.config = config
        self._wrapped_nodes: Dict[str, Callable] = {}

    def attach(self, graph: Any) -> Any:
        """
        Attach middleware to a LangGraph StateGraph.

        This wraps all nodes in the graph with AirOS reliability.
        Supports both legacy LangGraph (nodes are plain callables) and
        modern LangGraph (nodes are StateNodeSpec with .runnable).

        Args:
            graph: LangGraph StateGraph instance

        Returns:
            The modified graph
        """
        if not hasattr(graph, "nodes"):
            return graph

        for name, node in graph.nodes.items():
            if name in self._wrapped_nodes:
                continue

            # Modern LangGraph: nodes are StateNodeSpec dataclass with .runnable
            if hasattr(node, "runnable") and hasattr(node.runnable, "func"):
                wrapped = self._wrap_node_spec(node, name)
                graph.nodes[name] = wrapped
                self._wrapped_nodes[name] = wrapped
            else:
                # Legacy LangGraph: nodes are plain callables
                wrapped = self._wrap_node(node, name)
                graph.nodes[name] = wrapped
                self._wrapped_nodes[name] = wrapped

        return graph

    def _wrap_node_spec(self, node_spec: Any, name: str) -> Any:
        """
        Wrap a modern LangGraph StateNodeSpec by replacing its runnable's func.

        StateNodeSpec is a dataclass containing a RunnableCallable.
        We wrap the underlying func and reconstruct both.
        """
        from dataclasses import replace as dc_replace

        runnable = node_spec.runnable
        original_func = runnable.func

        wrapped_func = reliable_node(
            sentinel_schema=self.config.default_schema,
            llm_callable=self.config.llm_callable,
            fuse_limit=self.config.fuse_limit,
            node_name=name,
        )(original_func)

        # Reconstruct RunnableCallable with the wrapped func
        try:
            from langgraph._internal._runnable import RunnableCallable
            new_runnable = RunnableCallable(
                wrapped_func,
                name=runnable.name,
                tags=runnable.tags,
            )
        except ImportError:
            # Fallback: patch func directly
            runnable.func = wrapped_func
            return node_spec

        return dc_replace(node_spec, runnable=new_runnable)

    def _wrap_node(self, node: Callable, name: str) -> Callable:
        """Wrap a single node (legacy LangGraph — plain callable)."""
        return reliable_node(
            sentinel_schema=self.config.default_schema,
            llm_callable=self.config.llm_callable,
            fuse_limit=self.config.fuse_limit,
            node_name=name
        )(node)

    def wrap(
        self,
        schema: Optional[Type[BaseModel]] = None,
        name: Optional[str] = None
    ) -> Callable:
        """
        Decorator for manually wrapping specific nodes.

        Usage:
            @middleware.wrap(schema=OutputSchema)
            def my_node(state):
                return {"result": "value"}
        """
        def decorator(func: Callable) -> Callable:
            return reliable_node(
                sentinel_schema=schema or self.config.default_schema,
                llm_callable=self.config.llm_callable,
                fuse_limit=self.config.fuse_limit,
                node_name=name or func.__name__
            )(func)
        return decorator


# Register the adapter
AdapterRegistry.register("langgraph", LangGraphAdapter)
