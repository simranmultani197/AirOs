"""
CrewAI adapter for AgentCircuit.

Provides integration with CrewAI's agents, tasks, and crews.
"""
import functools
from typing import Any, Callable, Dict, List, Optional, Type
import time

from pydantic import BaseModel

from .base import FrameworkAdapter, AdapterConfig, AdapterRegistry
from ..fuse import Fuse, LoopError
from ..medic import Medic, MedicError
from ..sentinel import Sentinel, SentinelError
from ..storage import Storage


class CrewAIAdapter(FrameworkAdapter):
    """
    Adapter for CrewAI framework.

    Supports:
    - Agent wrapping
    - Task wrapping
    - Crew-level middleware
    - Tool wrapping
    """

    @property
    def framework_name(self) -> str:
        return "crewai"

    def wrap_node(
        self,
        func: Callable,
        schema: Optional[Type[BaseModel]] = None,
        node_name: Optional[str] = None,
        **kwargs
    ) -> Callable:
        """Wrap a CrewAI component."""
        return self._create_wrapper(func, schema, node_name)

    def create_middleware(self) -> "CrewAIMiddleware":
        """Create middleware for CrewAI."""
        return CrewAIMiddleware(self.config)

    def extract_state(self, *args, **kwargs) -> Dict[str, Any]:
        """Extract state from CrewAI task/agent execution."""
        state = {}

        # CrewAI Task input
        if args:
            first_arg = args[0]
            if hasattr(first_arg, "description"):
                # This is likely a Task
                state["task_description"] = first_arg.description
                if hasattr(first_arg, "context"):
                    state["context"] = first_arg.context
            elif isinstance(first_arg, dict):
                state = first_arg
            elif isinstance(first_arg, str):
                state["input"] = first_arg

        # Check kwargs
        state.update({
            k: v for k, v in kwargs.items()
            if k in ["input", "context", "task", "query"]
        })

        return state

    def extract_config(self, *args, **kwargs) -> Dict[str, Any]:
        """Extract config from CrewAI execution."""
        config = {}

        # Check for crew context
        if "crew" in kwargs:
            crew = kwargs["crew"]
            if hasattr(crew, "id"):
                config["run_id"] = str(crew.id)

        # Check for task ID
        for arg in args:
            if hasattr(arg, "id"):
                config.setdefault("configurable", {})["task_id"] = str(arg.id)

        return config

    def _create_wrapper(
        self,
        func: Callable,
        schema: Optional[Type[BaseModel]],
        node_name: Optional[str]
    ) -> Callable:
        """Create a wrapper for CrewAI components."""
        actual_name = node_name or getattr(func, "__name__", "crewai_node")

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
                # Convert CrewAI output to dict if needed
                if hasattr(result, "raw"):
                    result_dict = {"raw": result.raw}
                    if hasattr(result, "json_dict"):
                        result_dict.update(result.json_dict or {})
                elif isinstance(result, str):
                    result_dict = {"output": result}
                else:
                    result_dict = result if isinstance(result, dict) else {"output": result}

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

    @staticmethod
    def _safe_setattr(obj: Any, name: str, value: Any) -> None:
        """
        Set an attribute on an object, bypassing Pydantic restrictions.

        CrewAI v1.x Agent/Task are Pydantic BaseModel instances which block
        setting attributes that aren't declared fields. We use
        object.__setattr__ to bypass this restriction when patching methods.
        """
        try:
            setattr(obj, name, value)
        except (ValueError, AttributeError):
            # Pydantic model — bypass field validation
            object.__setattr__(obj, name, value)

    def wrap_agent(self, agent: Any, schema: Optional[Type[BaseModel]] = None) -> Any:
        """
        Wrap a CrewAI Agent with reliability features.

        Args:
            agent: CrewAI Agent instance
            schema: Optional output schema

        Returns:
            Wrapped agent
        """
        if hasattr(agent, "execute_task"):
            original_execute = agent.execute_task
            wrapper = self._create_wrapper(
                original_execute,
                schema,
                f"agent_{agent.role}" if hasattr(agent, "role") else "crewai_agent"
            )
            self._safe_setattr(agent, "execute_task", wrapper)
        return agent

    def wrap_task(self, task: Any, schema: Optional[Type[BaseModel]] = None) -> Any:
        """
        Wrap a CrewAI Task with reliability features.

        Supports both CrewAI v0.x (task.execute) and v1.x (task.execute_sync).

        Args:
            task: CrewAI Task instance
            schema: Optional output schema (can also use task.output_pydantic)

        Returns:
            Wrapped task
        """
        # Use task's output_pydantic if no schema provided
        if schema is None and hasattr(task, "output_pydantic"):
            schema = task.output_pydantic

        # CrewAI v1.x uses execute_sync; v0.x uses execute
        method_name = "execute_sync" if hasattr(task, "execute_sync") else "execute"

        if hasattr(task, method_name):
            original_execute = getattr(task, method_name)
            wrapper = self._create_wrapper(
                original_execute,
                schema,
                f"task_{task.description[:30]}" if hasattr(task, "description") else "crewai_task"
            )
            self._safe_setattr(task, method_name, wrapper)
        return task


class CrewAIMiddleware:
    """
    Middleware for automatically wrapping CrewAI components.

    Usage:
        from agentcircuit.adapters import CrewAIAdapter
        from crewai import Agent, Task, Crew

        adapter = CrewAIAdapter()
        middleware = adapter.create_middleware()

        # Wrap a crew
        crew = middleware.wrap_crew(my_crew)

        # Or wrap individual agents
        agent = middleware.wrap_agent(my_agent)
    """

    def __init__(self, config: AdapterConfig):
        self.config = config
        self._adapter = CrewAIAdapter(config)

    def wrap_agent(self, agent: Any, schema: Optional[Type[BaseModel]] = None) -> Any:
        """Wrap a CrewAI agent."""
        return self._adapter.wrap_agent(agent, schema)

    def wrap_task(self, task: Any, schema: Optional[Type[BaseModel]] = None) -> Any:
        """Wrap a CrewAI task."""
        return self._adapter.wrap_task(task, schema)

    def wrap_crew(self, crew: Any) -> Any:
        """
        Wrap an entire CrewAI Crew.

        Wraps all agents and tasks in the crew.

        Args:
            crew: CrewAI Crew instance

        Returns:
            Wrapped crew
        """
        if hasattr(crew, "agents"):
            for i, agent in enumerate(crew.agents):
                crew.agents[i] = self.wrap_agent(agent)

        if hasattr(crew, "tasks"):
            for i, task in enumerate(crew.tasks):
                crew.tasks[i] = self.wrap_task(task)

        return crew

    def agent(
        self,
        schema: Optional[Type[BaseModel]] = None,
        name: Optional[str] = None
    ) -> Callable:
        """Decorator for agent execution methods."""
        def decorator(func: Callable) -> Callable:
            return self._adapter.wrap_node(func, schema, name or func.__name__)
        return decorator

    def task(
        self,
        schema: Optional[Type[BaseModel]] = None,
        name: Optional[str] = None
    ) -> Callable:
        """Decorator for task execution methods."""
        return self.agent(schema, name)


# Register the adapter
AdapterRegistry.register("crewai", CrewAIAdapter)
