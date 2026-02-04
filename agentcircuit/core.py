import functools
import inspect
import time
from typing import Any, Optional, Callable, Dict, Type, Union
from pydantic import BaseModel

from .fuse import Fuse, LoopError
from .medic import Medic, MedicError
from .sentinel import Sentinel, SentinelError
from .storage import get_default_storage, BaseStorage
from .budget import BudgetFuse, TimeoutFuse, GlobalBudget
from .errors import BudgetExceededError, TimeoutExceededError
from .pricing import CostCalculator, estimate_tokens as _estimate_tokens


def reliable_node(
    sentinel_schema: Optional[Type[BaseModel]] = None,
    medic_repair: Optional[Callable[[Exception, Any], Any]] = None,
    llm_callable: Optional[Callable[[str], str]] = None,
    fuse_limit: int = 3,
    node_name: Optional[str] = None,
    storage: Optional[BaseStorage] = None,
    max_cost_usd: Optional[float] = None,
    max_seconds: Optional[float] = None,
    budget: Optional[GlobalBudget] = None,
    cost_per_token: Optional[float] = None,
    model: Optional[str] = None,
):
    """
    Decorator to make any AI agent node reliable.
    Integrates Fuse (loop detection), Medic (recovery), and Sentinel (validation).

    Args:
        sentinel_schema: Pydantic model to validate outputs against
        medic_repair: Legacy callback for custom repair logic
        llm_callable: LLM callable for intelligent error repair
        fuse_limit: Max identical states before tripping loop detection (default 3)
        node_name: Override the node name (defaults to function name)
        storage: Custom storage backend (defaults to in-memory)
        max_cost_usd: Maximum dollar cost for this node's run before tripping
        max_seconds: Maximum execution time in seconds before tripping
        budget: Shared GlobalBudget instance for cross-node cost/time limits
        cost_per_token: Override cost per token (USD). Overrides model pricing lookup.
        model: Model name for pricing table lookup (e.g. "gpt-4o", "claude-3-5-sonnet")
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract state and config
            state = args[0] if args else kwargs.get("state")
            config = None

            # Try to find config in args or kwargs
            for arg in args:
                if isinstance(arg, dict) and "configurable" in arg:
                    config = arg
                    break
            if not config:
                config = kwargs.get("config", {})

            # Attempt to identify run/thread
            run_id = (
                config.get("configurable", {}).get("thread_id")
                or config.get("run_id")
                or "local_dev_run"
            )

            actual_node_name = node_name or func.__name__

            # Initialize Components - use in-memory storage by default
            _storage = storage or get_default_storage()
            fuse = Fuse(limit=fuse_limit)
            medic = Medic(llm_callable=llm_callable)
            sentinel = Sentinel(schema=sentinel_schema)

            # Initialize cost/time circuit breakers
            _budget_fuse = BudgetFuse(max_cost_usd) if max_cost_usd else None
            _timeout_fuse = TimeoutFuse(max_seconds) if max_seconds else None

            # Cost calculator — priority: user override > model lookup > storage setting > default
            _model_name = model
            if not _model_name and hasattr(llm_callable, 'config'):
                _model_name = getattr(llm_callable.config, 'model_id', None)
            if not _model_name and hasattr(llm_callable, '_last_provider'):
                lp = getattr(llm_callable, '_last_provider', None)
                if lp and hasattr(lp, 'config'):
                    _model_name = getattr(lp.config, 'model_id', None)

            _cost_per_token = cost_per_token
            if _cost_per_token is None and not _model_name:
                cpt_str = _storage.get_setting("cost_per_token")
                if cpt_str:
                    _cost_per_token = float(cpt_str)

            _calculator = CostCalculator(model=_model_name, cost_per_token=_cost_per_token)

            # 0. Pre-execution budget checks
            # Check global budget before even starting
            if budget:
                budget.check_cost()
                budget.check_time()

            # Check per-node budget against cumulative run cost
            if _budget_fuse:
                current_run_cost = _storage.get_run_cost(run_id)
                _budget_fuse.check(current_run_cost)

            # 1. Fuse Check
            history_rows = _storage.get_run_history(run_id)
            node_history = [
                h["input_state"]
                for h in history_rows
                if h["node_id"] == actual_node_name
            ]

            try:
                fuse.check(history=[fuse._hash_state(s) for s in node_history], current_state=state)
            except LoopError as e:
                _storage.log_trace(
                    run_id=run_id,
                    node_id=actual_node_name,
                    input_state=state,
                    output_state=None,
                    status="failed_loop",
                    recovery_attempts=0
                )
                raise e

            # Filter kwargs for the wrapped function
            sig = inspect.signature(func)
            if "config" not in sig.parameters:
                kwargs.pop("config", None)

            # 2. Execution & Medic
            result = None
            status = "success"
            recovery_count = 0
            current_error = None
            saved_cost = 0.0
            diagnosis = None
            start_time = time.time()

            # Initial Execution
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                current_error = e
                diagnosis = str(e)
                if medic_repair and not llm_callable:
                    try:
                        result = medic_repair(e, state)
                        status = "repaired"
                        recovery_count = 1
                        current_error = None
                    except Exception as legacy_e:
                        current_error = legacy_e
                        diagnosis = str(legacy_e)

            # Validate result if no error
            if not current_error:
                try:
                    result = sentinel.validate(result)
                except SentinelError as se:
                    current_error = se
                    diagnosis = str(se)

            # Recovery Loop (up to 2 attempts)
            while current_error and recovery_count < 2:
                recovery_count += 1
                try:
                    raw_output = result if isinstance(current_error, SentinelError) else "N/A (Execution Failed)"

                    fixed_data = medic.attempt_recovery(
                        error=current_error,
                        input_state=state,
                        raw_output=raw_output,
                        node_id=actual_node_name,
                        recovery_attempts=recovery_count,
                        schema=sentinel_schema
                    )

                    result = sentinel.validate(fixed_data)
                    current_error = None
                    status = "repaired"

                    cost_to_reach_here = _storage.get_run_cost(run_id)
                    medic_input = _estimate_tokens(state) + _estimate_tokens(current_error) + 100
                    medic_output = _estimate_tokens(fixed_data)
                    medic_cost = _calculator.calculate(medic_input, medic_output)
                    raw_savings = cost_to_reach_here - medic_cost
                    saved_cost = max(0.0, raw_savings)

                except Exception as retry_e:
                    current_error = retry_e
                    diagnosis = str(retry_e)

            if current_error:
                duration_ms = (time.time() - start_time) * 1000
                _storage.log_trace(
                    run_id=run_id,
                    node_id=actual_node_name,
                    input_state=state,
                    output_state=str(current_error),
                    status="failed",
                    recovery_attempts=recovery_count,
                    saved_cost=0.0,
                    diagnosis=diagnosis,
                    duration_ms=duration_ms
                )
                raise current_error

            # Log Success — use CostCalculator for accurate pricing
            token_usage, estimated_cost = _calculator.estimate_from_objects(state, result)

            final_diagnosis = None
            if status == "repaired" and diagnosis:
                final_diagnosis = diagnosis
            elif status == "failed" and current_error:
                final_diagnosis = str(current_error)

            duration_ms = (time.time() - start_time) * 1000

            _storage.log_trace(
                run_id=run_id,
                node_id=actual_node_name,
                input_state=state,
                output_state=result,
                status=status,
                recovery_attempts=recovery_count,
                saved_cost=saved_cost,
                token_usage=token_usage,
                estimated_cost=estimated_cost,
                diagnosis=final_diagnosis,
                duration_ms=duration_ms
            )

            # Post-execution budget recording and checks
            if budget:
                budget.record_cost(estimated_cost)

            # Post-execution per-node budget check (after cost is logged)
            if _budget_fuse:
                updated_run_cost = _storage.get_run_cost(run_id)
                _budget_fuse.check(updated_run_cost)

            # Post-execution timeout check
            if _timeout_fuse:
                _timeout_fuse.check(start_time)

            # Post-execution global budget checks
            if budget:
                budget.check_cost()
                budget.check_time()

            return result
        return wrapper
    return decorator


# Clean alias - the primary public API
reliable = reliable_node
