"""
AutoGen adapter for AgentCircuit.

Provides integration with Microsoft AutoGen's conversable agents and group chats.
"""
import functools
from typing import Any, Callable, Dict, List, Optional, Type, Union
import time

from pydantic import BaseModel

from .base import FrameworkAdapter, AdapterConfig, AdapterRegistry
from ..fuse import Fuse, LoopError
from ..medic import Medic, MedicError
from ..sentinel import Sentinel, SentinelError
from ..storage import Storage


class AutoGenAdapter(FrameworkAdapter):
    """
    Adapter for Microsoft AutoGen framework.

    Supports:
    - ConversableAgent wrapping
    - AssistantAgent wrapping
    - UserProxyAgent wrapping
    - GroupChat middleware
    - Function/Tool registration
    """

    @property
    def framework_name(self) -> str:
        return "autogen"

    def wrap_node(
        self,
        func: Callable,
        schema: Optional[Type[BaseModel]] = None,
        node_name: Optional[str] = None,
        **kwargs
    ) -> Callable:
        """Wrap an AutoGen component."""
        return self._create_wrapper(func, schema, node_name)

    def create_middleware(self) -> "AutoGenMiddleware":
        """Create middleware for AutoGen."""
        return AutoGenMiddleware(self.config)

    def extract_state(self, *args, **kwargs) -> Dict[str, Any]:
        """Extract state from AutoGen message/execution."""
        state = {}

        # AutoGen message structure
        if args:
            first_arg = args[0]
            if isinstance(first_arg, dict):
                state = first_arg
            elif isinstance(first_arg, str):
                state["content"] = first_arg
            elif hasattr(first_arg, "content"):
                state["content"] = first_arg.content

        # Common AutoGen kwargs
        for key in ["message", "messages", "sender", "recipient", "context"]:
            if key in kwargs:
                value = kwargs[key]
                if hasattr(value, "name"):
                    state[key] = value.name
                elif isinstance(value, list):
                    state[key] = [
                        m if isinstance(m, dict) else {"content": str(m)}
                        for m in value
                    ]
                else:
                    state[key] = value

        return state

    def extract_config(self, *args, **kwargs) -> Dict[str, Any]:
        """Extract config from AutoGen execution."""
        config = {"configurable": {}}

        # Check for chat ID or session
        if "chat_id" in kwargs:
            config["run_id"] = kwargs["chat_id"]
        elif "session_id" in kwargs:
            config["run_id"] = kwargs["session_id"]

        # Check sender/recipient for context
        if "sender" in kwargs and hasattr(kwargs["sender"], "name"):
            config["configurable"]["sender"] = kwargs["sender"].name
        if "recipient" in kwargs and hasattr(kwargs["recipient"], "name"):
            config["configurable"]["recipient"] = kwargs["recipient"].name

        return config

    def _create_wrapper(
        self,
        func: Callable,
        schema: Optional[Type[BaseModel]],
        node_name: Optional[str]
    ) -> Callable:
        """Create a wrapper for AutoGen components."""
        actual_name = node_name or getattr(func, "__name__", "autogen_node")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            storage = Storage(db_path=self.config.db_path) if self.config.storage_enabled else None
            fuse = Fuse(limit=self.config.fuse_limit) if self.config.fuse_enabled else None
            medic = Medic(
                llm_callable=self.config.llm_callable,
                model=self.config.model,
                fallback_models=self.config.fallback_models
            ) if self.config.medic_enabled else None
            sentinel = Sentinel(schema=schema) if self.config.sentinel_enabled and schema else None

            state = self.extract_state(*args, **kwargs)
            config = self.extract_config(*args, **kwargs)
            run_id = self.get_run_id(config)

            start_time = time.time()
            result = None
            status = "success"
            recovery_count = 0
            diagnosis = None
            current_error = None

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

            # Execute
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                current_error = e
                diagnosis = str(e)

            # Validate
            if not current_error and sentinel:
                # Convert AutoGen output to dict
                if isinstance(result, str):
                    result_dict = {"content": result}
                elif isinstance(result, tuple):
                    # AutoGen often returns (terminate, response) tuples
                    result_dict = {
                        "terminate": result[0] if len(result) > 0 else False,
                        "content": result[1] if len(result) > 1 else None
                    }
                elif isinstance(result, dict):
                    result_dict = result
                else:
                    result_dict = {"output": result}

                try:
                    validated = sentinel.validate(result_dict)
                    result = validated
                except SentinelError as e:
                    current_error = e
                    diagnosis = str(e)

            # Recovery
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

            # Log
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

            output_state = result if isinstance(result, dict) else {"output": str(result)}
            if storage:
                storage.log_trace(
                    run_id=run_id,
                    node_id=actual_name,
                    input_state=state,
                    output_state=output_state,
                    status=status,
                    recovery_attempts=recovery_count,
                    diagnosis=diagnosis,
                    duration_ms=duration_ms
                )

            return result

        return wrapper

    def wrap_agent(
        self,
        agent: Any,
        schema: Optional[Type[BaseModel]] = None
    ) -> Any:
        """
        Wrap an AutoGen agent with reliability features.

        Args:
            agent: AutoGen ConversableAgent or subclass
            schema: Optional output schema

        Returns:
            Wrapped agent
        """
        agent_name = getattr(agent, "name", "autogen_agent")

        # Wrap the generate_reply method
        if hasattr(agent, "generate_reply"):
            original_generate = agent.generate_reply
            agent.generate_reply = self._create_wrapper(
                original_generate,
                schema,
                f"{agent_name}_generate_reply"
            )

        # Wrap registered functions
        if hasattr(agent, "_function_map"):
            for func_name, func in agent._function_map.items():
                agent._function_map[func_name] = self._create_wrapper(
                    func,
                    None,
                    f"{agent_name}_{func_name}"
                )

        return agent

    def wrap_function(
        self,
        func: Callable,
        schema: Optional[Type[BaseModel]] = None,
        name: Optional[str] = None
    ) -> Callable:
        """
        Wrap a function for registration with AutoGen agents.

        Args:
            func: Function to wrap
            schema: Optional output schema
            name: Optional custom name

        Returns:
            Wrapped function
        """
        return self._create_wrapper(func, schema, name or func.__name__)


class AutoGenMiddleware:
    """
    Middleware for automatically wrapping AutoGen components.

    Usage:
        from agentcircuit.adapters import AutoGenAdapter
        import autogen

        adapter = AutoGenAdapter()
        middleware = adapter.create_middleware()

        # Wrap an agent
        agent = middleware.wrap_agent(my_agent)

        # Wrap a group chat
        group_chat = middleware.wrap_group_chat(my_group_chat)

        # Use as decorator for functions
        @middleware.function(schema=MySchema)
        def my_function(input: str) -> str:
            return "result"
    """

    def __init__(self, config: AdapterConfig):
        self.config = config
        self._adapter = AutoGenAdapter(config)

    def wrap_agent(
        self,
        agent: Any,
        schema: Optional[Type[BaseModel]] = None
    ) -> Any:
        """Wrap an AutoGen agent."""
        return self._adapter.wrap_agent(agent, schema)

    def wrap_group_chat(self, group_chat: Any) -> Any:
        """
        Wrap an AutoGen GroupChat.

        Wraps all agents in the group chat.

        Args:
            group_chat: AutoGen GroupChat instance

        Returns:
            Wrapped group chat
        """
        if hasattr(group_chat, "agents"):
            for i, agent in enumerate(group_chat.agents):
                group_chat.agents[i] = self.wrap_agent(agent)

        return group_chat

    def wrap_function(
        self,
        func: Callable,
        schema: Optional[Type[BaseModel]] = None,
        name: Optional[str] = None
    ) -> Callable:
        """Wrap a function for use with AutoGen."""
        return self._adapter.wrap_function(func, schema, name)

    def function(
        self,
        schema: Optional[Type[BaseModel]] = None,
        name: Optional[str] = None
    ) -> Callable:
        """Decorator for wrapping functions."""
        def decorator(func: Callable) -> Callable:
            return self._adapter.wrap_function(func, schema, name)
        return decorator

    def agent_reply(
        self,
        schema: Optional[Type[BaseModel]] = None,
        name: Optional[str] = None
    ) -> Callable:
        """Decorator for agent reply methods."""
        return self.function(schema, name)


# Register the adapter
AdapterRegistry.register("autogen", AutoGenAdapter)
