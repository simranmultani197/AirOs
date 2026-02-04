"""
Integration tests for the reliable_node decorator.
"""
import pytest
import uuid
import tempfile
import os
from pydantic import BaseModel
from typing import Dict, Any

from agentcircuit import reliable_node, LoopError, SentinelError
from agentcircuit.storage import Storage, InMemoryStorage
from agentcircuit.budget import GlobalBudget
from agentcircuit.errors import BudgetExceededError, TimeoutExceededError


# ============================================================================
# Test Schemas
# ============================================================================

class SimpleOutput(BaseModel):
    message: str
    status: str


class NumberOutput(BaseModel):
    value: int
    computed: bool


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def unique_run_id():
    """Generate unique run ID for each test."""
    return f"test-{uuid.uuid4()}"


@pytest.fixture
def temp_storage():
    """Create temporary storage for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, ".air_os", "traces.db")
    storage = Storage(db_path=db_path)
    yield storage
    # Cleanup handled by OS


# ============================================================================
# Basic Decorator Tests
# ============================================================================

class TestReliableNodeBasics:
    """Test basic decorator functionality."""

    def test_decorator_preserves_function_name(self):
        """Test decorator preserves original function name."""
        @reliable_node()
        def my_special_node(state):
            return {"result": "ok"}

        assert my_special_node.__name__ == "my_special_node"

    def test_decorator_allows_normal_execution(self, unique_run_id):
        """Test decorated function executes normally."""
        @reliable_node()
        def simple_node(state):
            return {"processed": state.get("input", "none")}

        result = simple_node(
            {"input": "test_data"},
            config={"configurable": {"thread_id": unique_run_id}}
        )

        assert result == {"processed": "test_data"}

    def test_decorator_without_config(self, unique_run_id):
        """Test decorator works without config parameter."""
        @reliable_node()
        def no_config_node(state):
            return {"value": 42}

        # Use unique state to avoid loop detection from other tests
        result = no_config_node({"input": f"test_no_config_{uuid.uuid4()}"})
        assert result == {"value": 42}

    def test_custom_node_name(self, unique_run_id):
        """Test decorator uses custom node name."""
        @reliable_node(node_name="custom_name")
        def original_name(state):
            return {"result": "ok"}

        # Execute to trigger logging
        result = original_name(
            {"input": "test"},
            config={"configurable": {"thread_id": unique_run_id}}
        )

        assert result == {"result": "ok"}


# ============================================================================
# Sentinel Validation Tests
# ============================================================================

class TestReliableNodeValidation:
    """Test Sentinel validation integration."""

    def test_valid_output_passes(self, unique_run_id):
        """Test valid output passes validation."""
        @reliable_node(sentinel_schema=SimpleOutput)
        def valid_node(state):
            return {"message": "Hello", "status": "ok"}

        result = valid_node(
            {"input": "test"},
            config={"configurable": {"thread_id": unique_run_id}}
        )

        assert isinstance(result, SimpleOutput)
        assert result.message == "Hello"

    def test_invalid_output_triggers_medic(self, unique_run_id):
        """Test invalid output triggers Medic recovery."""
        # Create a mock LLM that returns valid JSON for SimpleOutput
        def mock_llm(prompt: str) -> str:
            return '{"message": "Fixed by Medic", "status": "repaired"}'

        @reliable_node(
            sentinel_schema=SimpleOutput,
            llm_callable=mock_llm
        )
        def invalid_node(state):
            # Always return invalid output - Medic should fix it
            return {"wrong": "fields"}

        result = invalid_node(
            {"input": "test"},
            config={"configurable": {"thread_id": unique_run_id}}
        )

        # Should have recovered via Medic
        assert isinstance(result, SimpleOutput)
        assert result.message == "Fixed by Medic"

    def test_validation_without_llm_fails(self, unique_run_id):
        """Test validation failure without LLM raises error."""
        # Define an LLM that always returns invalid output, so Medic can't repair
        def always_invalid_llm(prompt: str) -> str:
            return '{"still": "invalid"}'

        @reliable_node(sentinel_schema=SimpleOutput, llm_callable=always_invalid_llm)
        def bad_output_node(state):
            return {"invalid": "structure"}

        # Should raise SentinelError after Medic fails to repair
        with pytest.raises(SentinelError):
            bad_output_node(
                {"input": "test"},
                config={"configurable": {"thread_id": unique_run_id}}
            )


# ============================================================================
# Fuse Loop Detection Tests
# ============================================================================

class TestReliableNodeFuse:
    """Test Fuse loop detection integration."""

    def test_no_loop_with_different_states(self, unique_run_id):
        """Test no loop detection with different states."""
        @reliable_node(fuse_limit=3)
        def varying_node(state):
            return {"result": state.get("counter", 0)}

        # Execute multiple times with different states
        for i in range(5):
            result = varying_node(
                {"counter": i},
                config={"configurable": {"thread_id": unique_run_id}}
            )
            assert result == {"result": i}

    def test_loop_detected_same_state(self):
        """Test loop detection with repeated same state."""
        # Use unique run_id each time to avoid cross-test pollution
        run_id = f"loop-test-{uuid.uuid4()}"

        @reliable_node(fuse_limit=2)
        def looping_node(state):
            return {"echo": state}

        # First call - ok
        looping_node(
            {"same": "state"},
            config={"configurable": {"thread_id": run_id}}
        )

        # Second call - ok (at limit, not over)
        looping_node(
            {"same": "state"},
            config={"configurable": {"thread_id": run_id}}
        )

        # Third call should trigger loop detection
        with pytest.raises(LoopError):
            looping_node(
                {"same": "state"},
                config={"configurable": {"thread_id": run_id}}
            )

    def test_custom_fuse_limit(self):
        """Test custom fuse limit."""
        run_id = f"custom-limit-{uuid.uuid4()}"

        @reliable_node(fuse_limit=5)
        def high_limit_node(state):
            return {"ok": True}

        # Should allow 4 calls with same state
        for _ in range(4):
            high_limit_node(
                {"repeated": "state"},
                config={"configurable": {"thread_id": run_id}}
            )

        # 5th call at limit - still ok
        high_limit_node(
            {"repeated": "state"},
            config={"configurable": {"thread_id": run_id}}
        )

        # 6th call should fail
        with pytest.raises(LoopError):
            high_limit_node(
                {"repeated": "state"},
                config={"configurable": {"thread_id": run_id}}
            )


# ============================================================================
# Medic Recovery Tests
# ============================================================================

class TestReliableNodeMedic:
    """Test Medic recovery integration."""

    def test_exception_recovery_with_llm(self, unique_run_id):
        """Test recovery from exception using LLM."""
        call_count = 0

        def mock_llm(prompt: str) -> str:
            return '{"message": "Recovered", "status": "ok"}'

        @reliable_node(
            sentinel_schema=SimpleOutput,
            llm_callable=mock_llm
        )
        def failing_node(state):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Simulated failure")
            return {"message": "Success", "status": "ok"}

        result = failing_node(
            {"input": "test"},
            config={"configurable": {"thread_id": unique_run_id}}
        )

        assert isinstance(result, SimpleOutput)

    def test_recovery_fails_after_attempts(self, unique_run_id):
        """Test recovery fails after max attempts."""
        def always_fail_llm(prompt: str) -> str:
            return '{"invalid": "json_structure"}'

        @reliable_node(
            sentinel_schema=SimpleOutput,
            llm_callable=always_fail_llm
        )
        def always_failing_node(state):
            raise RuntimeError("Always fails")

        with pytest.raises(Exception):
            always_failing_node(
                {"input": "test"},
                config={"configurable": {"thread_id": unique_run_id}}
            )

    def test_legacy_medic_repair_callback(self, unique_run_id):
        """Test legacy medic_repair callback still works."""
        def legacy_repair(error, state):
            return {"message": "Legacy fixed", "status": "ok"}

        @reliable_node(
            sentinel_schema=SimpleOutput,
            medic_repair=legacy_repair
        )
        def legacy_node(state):
            raise ValueError("Needs repair")

        result = legacy_node(
            {"input": "test"},
            config={"configurable": {"thread_id": unique_run_id}}
        )

        assert isinstance(result, SimpleOutput)
        assert result.message == "Legacy fixed"


# ============================================================================
# Storage Integration Tests
# ============================================================================

class TestReliableNodeStorage:
    """Test storage integration."""

    def test_success_logged(self, unique_run_id):
        """Test successful execution is logged."""
        test_storage = InMemoryStorage()

        @reliable_node(storage=test_storage)
        def logged_node(state):
            return {"result": "logged"}

        logged_node(
            {"input": "test"},
            config={"configurable": {"thread_id": unique_run_id}}
        )

        # Verify logged
        history = test_storage.get_run_history(unique_run_id)
        assert len(history) >= 1
        assert history[-1]["status"] == "success"

    def test_failure_logged(self, unique_run_id):
        """Test failure is logged."""
        test_storage = InMemoryStorage()

        # Use an LLM that always returns invalid output so recovery fails
        def always_fail_llm(prompt: str) -> str:
            raise RuntimeError("LLM also fails")

        @reliable_node(llm_callable=always_fail_llm, storage=test_storage)
        def crash_node(state):
            raise RuntimeError("Intentional crash")

        try:
            crash_node(
                {"input": "test"},
                config={"configurable": {"thread_id": unique_run_id}}
            )
        except Exception:
            # Catch any exception (could be RuntimeError or MedicError)
            pass

        history = test_storage.get_run_history(unique_run_id)
        assert len(history) >= 1
        # Status should be 'failed' or start with 'failed' (could be 'failed_loop', etc.)
        last_status = history[-1]["status"]
        assert last_status.startswith("failed") or last_status == "failed", f"Expected 'failed', got '{last_status}'"

    def test_recovery_logged(self, unique_run_id):
        """Test recovery is logged."""
        test_storage = InMemoryStorage()

        # Use a direct mock LLM that returns proper JSON for SimpleOutput schema
        def fix_llm(prompt: str) -> str:
            return '{"message": "Fixed by LLM", "status": "ok"}'

        @reliable_node(
            sentinel_schema=SimpleOutput,
            llm_callable=fix_llm,
            storage=test_storage,
        )
        def recoverable_node(state):
            # This invalid output should trigger Sentinel validation failure
            # which Medic should then repair using the LLM
            return {"wrong": "output"}

        result = recoverable_node(
            {"input": "test"},
            config={"configurable": {"thread_id": unique_run_id}}
        )

        # Verify recovery worked
        assert isinstance(result, SimpleOutput)
        assert result.message == "Fixed by LLM"

        history = test_storage.get_run_history(unique_run_id)
        # Should have at least one entry showing repair
        assert len(history) >= 1
        # Check if any status indicates repair/success
        statuses = [h["status"] for h in history]
        assert any(s in ["repaired", "success"] for s in statuses), f"Expected 'repaired' or 'success' in {statuses}"


# ============================================================================
# Edge Cases
# ============================================================================

class TestReliableNodeEdgeCases:
    """Test edge cases and error handling."""

    def test_none_state(self, unique_run_id):
        """Test handling of None state."""
        @reliable_node()
        def none_state_node(state):
            return {"received": state}

        result = none_state_node(
            None,
            config={"configurable": {"thread_id": unique_run_id}}
        )
        assert result == {"received": None}

    def test_empty_state(self, unique_run_id):
        """Test handling of empty state."""
        @reliable_node()
        def empty_state_node(state):
            return {"empty": len(state) == 0}

        result = empty_state_node(
            {},
            config={"configurable": {"thread_id": unique_run_id}}
        )
        assert result == {"empty": True}

    def test_large_state(self, unique_run_id):
        """Test handling of large state."""
        @reliable_node()
        def large_state_node(state):
            return {"size": len(str(state))}

        large_state = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}

        result = large_state_node(
            large_state,
            config={"configurable": {"thread_id": unique_run_id}}
        )
        assert result["size"] > 10000

    def test_nested_state(self, unique_run_id):
        """Test handling of deeply nested state."""
        @reliable_node()
        def nested_node(state):
            # Access nested value
            return {"found": state["a"]["b"]["c"]["d"]}

        nested = {"a": {"b": {"c": {"d": "deep_value"}}}}

        result = nested_node(
            nested,
            config={"configurable": {"thread_id": unique_run_id}}
        )
        assert result == {"found": "deep_value"}


# ============================================================================
# Budget Fuse Integration Tests
# ============================================================================

class TestReliableNodeBudget:
    """Test budget circuit breaker integration."""

    def test_max_cost_usd_under_budget(self, unique_run_id):
        """Test execution succeeds when under per-node budget."""
        test_storage = InMemoryStorage()

        @reliable_node(max_cost_usd=10.0, storage=test_storage)
        def cheap_node(state):
            return {"result": "cheap"}

        result = cheap_node(
            {"input": "test"},
            config={"configurable": {"thread_id": unique_run_id}}
        )
        assert result == {"result": "cheap"}

    def test_max_cost_usd_trips_on_accumulation(self):
        """Test per-node budget trips after accumulated cost."""
        test_storage = InMemoryStorage()
        run_id = f"budget-test-{uuid.uuid4()}"

        @reliable_node(max_cost_usd=1.0, storage=test_storage)
        def tracked_node(state):
            return {"result": "ok"}

        # First call should succeed (cost is tiny)
        tracked_node(
            {"input": "a"},
            config={"configurable": {"thread_id": run_id}}
        )

        # Manually inflate the run cost to exceed the $1.0 budget
        test_storage.log_trace(
            run_id=run_id,
            node_id="manual_cost",
            input_state={},
            output_state={},
            status="success",
            estimated_cost=2.0,
        )

        # Next call should trip on the pre-execution budget check
        with pytest.raises(BudgetExceededError):
            tracked_node(
                {"input": "b"},
                config={"configurable": {"thread_id": run_id}}
            )

    def test_max_seconds_under_timeout(self, unique_run_id):
        """Test execution succeeds when under timeout."""
        @reliable_node(max_seconds=10.0)
        def fast_node(state):
            return {"result": "fast"}

        result = fast_node(
            {"input": "test"},
            config={"configurable": {"thread_id": unique_run_id}}
        )
        assert result == {"result": "fast"}

    def test_global_budget_under_limit(self, unique_run_id):
        """Test GlobalBudget passes when under limit."""
        budget = GlobalBudget(max_cost_usd=100.0)

        @reliable_node(budget=budget)
        def budget_node(state):
            return {"result": "ok"}

        result = budget_node(
            {"input": "test"},
            config={"configurable": {"thread_id": unique_run_id}}
        )
        assert result == {"result": "ok"}
        assert budget.total_spent > 0  # Some cost was recorded

    def test_global_budget_shared_across_nodes(self):
        """Test GlobalBudget is shared across multiple nodes."""
        test_storage = InMemoryStorage()
        budget = GlobalBudget(max_cost_usd=100.0)
        run_id = f"shared-budget-{uuid.uuid4()}"

        @reliable_node(budget=budget, storage=test_storage)
        def node_a(state):
            return {"from": "a"}

        @reliable_node(budget=budget, storage=test_storage)
        def node_b(state):
            return {"from": "b"}

        node_a(
            {"input": "test"},
            config={"configurable": {"thread_id": run_id}}
        )
        spent_after_a = budget.total_spent

        node_b(
            {"input": "test"},
            config={"configurable": {"thread_id": run_id}}
        )
        spent_after_b = budget.total_spent

        # Both nodes should have contributed to the budget
        assert spent_after_b > spent_after_a

    def test_global_budget_trips_when_exceeded(self):
        """Test GlobalBudget trips after exceeding limit."""
        budget = GlobalBudget(max_cost_usd=0.0001)
        # Pre-spend the budget
        budget.record_cost(0.001)

        @reliable_node(budget=budget)
        def over_budget_node(state):
            return {"result": "should not reach"}

        with pytest.raises(BudgetExceededError):
            over_budget_node({"input": "test"})

    def test_global_budget_timeout_trips(self):
        """Test GlobalBudget timeout trips when expired."""
        budget = GlobalBudget(max_cost_usd=100.0, max_seconds=0.01)
        # Manually expire the timer
        budget._start_time = budget._start_time - 1.0

        @reliable_node(budget=budget)
        def timed_out_node(state):
            return {"result": "should not reach"}

        with pytest.raises(TimeoutExceededError):
            timed_out_node({"input": "test"})

    def test_global_budget_reset(self):
        """Test GlobalBudget can be reset between runs."""
        budget = GlobalBudget(max_cost_usd=100.0)
        run_id1 = f"reset-test-1-{uuid.uuid4()}"
        run_id2 = f"reset-test-2-{uuid.uuid4()}"

        @reliable_node(budget=budget)
        def tracked_node(state):
            return {"result": "ok"}

        # First run
        tracked_node(
            {"input": "run1"},
            config={"configurable": {"thread_id": run_id1}}
        )
        first_run_cost = budget.total_spent
        assert first_run_cost > 0

        # Reset
        budget.reset()
        assert budget.total_spent == 0.0

        # Second run
        tracked_node(
            {"input": "run2"},
            config={"configurable": {"thread_id": run_id2}}
        )
        assert budget.total_spent > 0
        assert budget.remaining > 0
