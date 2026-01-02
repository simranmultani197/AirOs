import functools
import traceback
import inspect
from typing import Any, Optional, Callable, Dict
from pydantic import BaseModel

from .fuse import Fuse, LoopError
from .medic import Medic, MedicError
from .sentinel import Sentinel, SentinelError
from .storage import Storage

def reliable_node(
    sentinel_schema: Optional[BaseModel] = None,
    # Previously medic_repair, now we prefer llm_callable or just generic valid arg
    # We keep medic_repair for backward compat or custom direct repair, 
    # but we add llm_callable for the prompt-based repair.
    medic_repair: Optional[Callable[[Exception, Any], Any]] = None,
    llm_callable: Optional[Callable[[str], str]] = None,
    fuse_limit: int = 3,
    node_name: Optional[str] = None
):
    """
    Decorator to make a LangGraph node reliable.
    Integrates Fuse (loop detection), Medic (recovery), and Sentinel (validation).
    Logs all execution traces to local SQLite.
    """
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract state and config
            # LangGraph usually passes (state) or (state, config)
            # We need to handle both.
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
            # Prioritize thread_id (persistence thread) -> run_id -> generic
            run_id = (
                config.get("configurable", {}).get("thread_id") 
                or config.get("run_id") 
                or "local_dev_run"
            )
            
            # Determine Node Name
            actual_node_name = node_name or func.__name__

            # Initialize Components
            storage = Storage()
            fuse = Fuse(limit=fuse_limit)
            # Prioritize llm_callable for the new Medic logic, fallback to legacy repair_callback
            medic = Medic(llm_callable=llm_callable)
            # If the user passed the old style callback, we might need a bridge or just handle it separately?
            # For now, let's assume if llm_callable is passed, we use the new Medic loop.
            sentinel = Sentinel(schema=sentinel_schema)

            # Cost Tracking Helpers
            def estimate_tokens(obj: Any) -> int:
                # Heuristic: 4 chars ~= 1 token
                s = str(obj)
                return len(s) // 4
            
            def get_cost(tokens: int) -> float:
                price_str = storage.get_setting("cost_per_token") or "0.000005"
                return tokens * float(price_str)

            # 1. Fuse Check
            # Retrieve history for loop detection
            # ideally we check the actual state objects from history
            # For simplicity in Phase 1, we fetch past inputs for this run/thread
            # and check if we are looping on the EXACT same input state for this node.
            # (Looping means visiting the same node with same state multiple times)
            
            # Fetch recent traces for this run
            history_rows = storage.get_run_history(run_id)
            # Filter for this node? Or global loop?
            # Typically "Loop" means same node, same state.
            node_history = [
                h["input_state"] 
                for h in history_rows 
                if h["node_id"] == actual_node_name
            ]
            
            try:
                fuse.check(history=[fuse._hash_state(s) for s in node_history], current_state=state)
            except LoopError as e:
                # Log loop error
                storage.log_trace(
                    run_id=run_id,
                    node_id=actual_node_name,
                    input_state=state,
                    output_state=None,
                    status="failed_loop",
                    recovery_attempts=0
                )
                raise e

            # Filter kwargs for the wrapped function
            # If function doesn't accept 'config', remove it from kwargs
            sig = inspect.signature(func)
            if "config" not in sig.parameters:
                kwargs.pop("config", None)

            # 2. Execution & Medic
            # We wrap this in a loop for retries
            # But wait, Medic.attempt_recovery just does one fix?
            # The Requirement says: "If it fails after 2 attempts...". 
            # So we need a loop here.
            
            result = None
            status = "success"
            recovery_count = 0
            current_error = None
            saved_cost = 0.0
            diagnosis = None
            
            # Initial Execution
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                current_error = e
                diagnosis = str(e) # Capture diagnosis
                # Fallback to legacy medic_repair if provided and no llm_callable
                if medic_repair and not llm_callable:
                    try:
                        result = medic_repair(e, state)
                        status = "repaired"
                        recovery_count = 1
                        current_error = None
                    except Exception as legacy_e:
                        current_error = legacy_e
                        diagnosis = str(legacy_e)
                else:
                    # Proceed to Medic Loop
                    pass

            # If we have an error or if validation fails (checked below), we enter recovery loop.
            # Actually, validation is separate. 
            # The prompt says: "When a wrapped node throws an exception... Increment recovery_attempts... Capture raw error... invoke Repair LLM"
            
            # Also: "Take the output from the Medic and re-run the validation logic."
            
            # Let's verify result first if no exception
            if not current_error:
                try:
                    result = sentinel.validate(result)
                except SentinelError as se:
                    current_error = se
                    diagnosis = str(se)

            # Recovery Loop
            # We try up to 2 times
            while current_error and recovery_count < 2:
                recovery_count += 1
                try:
                    # Provide raw_output if available?
                    # If exception happened during func, raw_output might be unknown/None.
                    # If exception was SentinelError, result holds the invalid output.
                    raw_output = result if isinstance(current_error, SentinelError) else "N/A (Execution Failed)"
                    
                    fixed_data = medic.attempt_recovery(
                        error=current_error,
                        input_state=state,
                        raw_output=raw_output,
                        node_id=actual_node_name,
                        recovery_attempts=recovery_count
                    )
                    
                    # Re-validate
                    result = sentinel.validate(fixed_data)
                    
                    # If we got here, success!
                    current_error = None
                    status = "repaired"
                    
                    # Savings Calculation
                    # Cost to reach here = sum of all previous nodes in this run
                    cost_to_reach_here = storage.get_run_cost(run_id)
                    # Medic Cost (Estimated) - input state + error msg + prompt + fixed output
                    medic_tokens = estimate_tokens(state) + estimate_tokens(current_error) + estimate_tokens(fixed_data) + 100 # prompt overhead
                    medic_cost = get_cost(medic_tokens)
                    
                    # Net Savings: We saved the cost of re-running everything up to here, minus what we spent on Medic.
                    # ROI = Saved / Spent. 
                    # If we are early in the graph, savings might be small or negative (if medic is expensive).
                    # If deeper, savings are huge.
                    
                    # Logic: Avoid negative savings logging for UX. Or show truth? 
                    # "Money Saved" usually implies positive.
                    raw_savings = cost_to_reach_here - medic_cost
                    saved_cost = max(0.0, raw_savings)
                    
                except Exception as retry_e:
                    current_error = retry_e
                    diagnosis = str(retry_e) # Update diagnosis on retry failure
            
            if current_error:
                # Log failure
                storage.log_trace(
                    run_id=run_id,
                    node_id=actual_node_name,
                    input_state=state,
                    output_state=str(current_error),
                    status="failed",
                    recovery_attempts=recovery_count,
                    saved_cost=0.0,
                    diagnosis=diagnosis
                )
                print(f"CRITICAL FAILURE in {actual_node_name}: {current_error}")
                raise current_error

            # 3. Sentinel Check (Done inside loop/initial block now)
            # Just kept for structure but mostly handled above.

            # 4. Log Success
            # Calculate final costs
            # We treat the final result + input as the usage for this node step
            token_usage = estimate_tokens(state) + estimate_tokens(result)
            estimated_cost = get_cost(token_usage)
            
            # If we repaired, we also incurred Medic costs. 
            # Ideally we track Medic LLM usage separately, but for MVP we assume Medic output is part of result.
            # We already set saved_cost in the loop if repaired.
            
            # Capture the original error as diagnosis if we recovered
            final_diagnosis = None
            if status == "repaired" and diagnosis:
                final_diagnosis = diagnosis
            elif status == "failed" and current_error:
                final_diagnosis = str(current_error)

            storage.log_trace(
                run_id=run_id,
                node_id=actual_node_name,
                input_state=state,
                output_state=result,
                status=status,
                recovery_attempts=recovery_count,
                saved_cost=saved_cost,
                token_usage=token_usage,
                estimated_cost=estimated_cost,
                diagnosis=final_diagnosis
            )

            return result
        return wrapper
    return decorator
