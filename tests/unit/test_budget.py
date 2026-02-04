"""
Unit tests for the Budget module - Cost-saving circuit breakers.
"""
import time
import threading
import pytest

from agentcircuit.budget import BudgetFuse, TimeoutFuse, GlobalBudget
from agentcircuit.errors import BudgetExceededError, TimeoutExceededError


# ============================================================================
# BudgetFuse Tests
# ============================================================================

class TestBudgetFuseBasics:
    """Test basic BudgetFuse functionality."""

    def test_init_valid(self):
        """Test BudgetFuse initializes with valid limit."""
        fuse = BudgetFuse(max_cost_usd=1.0)
        assert fuse.max_cost_usd == 1.0

    def test_init_small_budget(self):
        """Test BudgetFuse with very small budget."""
        fuse = BudgetFuse(max_cost_usd=0.001)
        assert fuse.max_cost_usd == 0.001

    def test_init_zero_raises(self):
        """Test BudgetFuse raises on zero budget."""
        with pytest.raises(ValueError, match="must be positive"):
            BudgetFuse(max_cost_usd=0)

    def test_init_negative_raises(self):
        """Test BudgetFuse raises on negative budget."""
        with pytest.raises(ValueError, match="must be positive"):
            BudgetFuse(max_cost_usd=-1.0)


class TestBudgetFuseCheck:
    """Test BudgetFuse check logic."""

    def test_under_budget_passes(self):
        """Test check passes when under budget."""
        fuse = BudgetFuse(max_cost_usd=1.0)
        fuse.check(0.5)  # Should not raise

    def test_zero_cost_passes(self):
        """Test check passes with zero cost."""
        fuse = BudgetFuse(max_cost_usd=1.0)
        fuse.check(0.0)  # Should not raise

    def test_at_limit_raises(self):
        """Test check raises when exactly at limit."""
        fuse = BudgetFuse(max_cost_usd=1.0)
        with pytest.raises(BudgetExceededError) as exc_info:
            fuse.check(1.0)
        assert exc_info.value.spent == 1.0
        assert exc_info.value.limit == 1.0

    def test_over_limit_raises(self):
        """Test check raises when over limit."""
        fuse = BudgetFuse(max_cost_usd=1.0)
        with pytest.raises(BudgetExceededError) as exc_info:
            fuse.check(1.5)
        assert exc_info.value.spent == 1.5
        assert "Budget exceeded" in str(exc_info.value)

    def test_just_under_limit_passes(self):
        """Test check passes when just under limit."""
        fuse = BudgetFuse(max_cost_usd=1.0)
        fuse.check(0.999999)  # Should not raise

    def test_error_has_spent_and_limit(self):
        """Test BudgetExceededError contains spent and limit."""
        fuse = BudgetFuse(max_cost_usd=5.0)
        with pytest.raises(BudgetExceededError) as exc_info:
            fuse.check(7.5)
        assert exc_info.value.spent == 7.5
        assert exc_info.value.limit == 5.0


# ============================================================================
# TimeoutFuse Tests
# ============================================================================

class TestTimeoutFuseBasics:
    """Test basic TimeoutFuse functionality."""

    def test_init_valid(self):
        """Test TimeoutFuse initializes with valid limit."""
        fuse = TimeoutFuse(max_seconds=30.0)
        assert fuse.max_seconds == 30.0

    def test_init_small_timeout(self):
        """Test TimeoutFuse with very small timeout."""
        fuse = TimeoutFuse(max_seconds=0.001)
        assert fuse.max_seconds == 0.001

    def test_init_zero_raises(self):
        """Test TimeoutFuse raises on zero timeout."""
        with pytest.raises(ValueError, match="must be positive"):
            TimeoutFuse(max_seconds=0)

    def test_init_negative_raises(self):
        """Test TimeoutFuse raises on negative timeout."""
        with pytest.raises(ValueError, match="must be positive"):
            TimeoutFuse(max_seconds=-1.0)


class TestTimeoutFuseCheck:
    """Test TimeoutFuse check logic."""

    def test_within_timeout_passes(self):
        """Test check passes when within timeout."""
        fuse = TimeoutFuse(max_seconds=10.0)
        start = time.time()
        fuse.check(start)  # Should not raise (just started)

    def test_expired_timeout_raises(self):
        """Test check raises when timeout expired."""
        fuse = TimeoutFuse(max_seconds=0.01)
        start = time.time() - 1.0  # 1 second ago
        with pytest.raises(TimeoutExceededError) as exc_info:
            fuse.check(start)
        assert exc_info.value.elapsed >= 1.0
        assert exc_info.value.limit == 0.01

    def test_error_has_elapsed_and_limit(self):
        """Test TimeoutExceededError contains elapsed and limit."""
        fuse = TimeoutFuse(max_seconds=5.0)
        start = time.time() - 10.0  # 10 seconds ago
        with pytest.raises(TimeoutExceededError) as exc_info:
            fuse.check(start)
        assert exc_info.value.elapsed >= 10.0
        assert exc_info.value.limit == 5.0
        assert "Timeout exceeded" in str(exc_info.value)


# ============================================================================
# GlobalBudget Tests
# ============================================================================

class TestGlobalBudgetBasics:
    """Test basic GlobalBudget functionality."""

    def test_init_cost_only(self):
        """Test GlobalBudget with cost limit only."""
        gb = GlobalBudget(max_cost_usd=5.0)
        assert gb.max_cost_usd == 5.0
        assert gb.max_seconds is None
        assert gb.total_spent == 0.0
        assert gb.remaining == 5.0

    def test_init_cost_and_time(self):
        """Test GlobalBudget with cost and time limits."""
        gb = GlobalBudget(max_cost_usd=5.0, max_seconds=60.0)
        assert gb.max_cost_usd == 5.0
        assert gb.max_seconds == 60.0

    def test_init_zero_cost_raises(self):
        """Test GlobalBudget raises on zero cost."""
        with pytest.raises(ValueError, match="max_cost_usd must be positive"):
            GlobalBudget(max_cost_usd=0)

    def test_init_negative_time_raises(self):
        """Test GlobalBudget raises on negative time."""
        with pytest.raises(ValueError, match="max_seconds must be positive"):
            GlobalBudget(max_cost_usd=5.0, max_seconds=-1.0)


class TestGlobalBudgetCostTracking:
    """Test GlobalBudget cost tracking."""

    def test_record_cost(self):
        """Test recording costs."""
        gb = GlobalBudget(max_cost_usd=5.0)
        gb.record_cost(1.0)
        assert gb.total_spent == 1.0
        assert gb.remaining == 4.0

    def test_record_multiple_costs(self):
        """Test recording multiple costs accumulates."""
        gb = GlobalBudget(max_cost_usd=5.0)
        gb.record_cost(1.0)
        gb.record_cost(0.5)
        gb.record_cost(2.0)
        assert gb.total_spent == 3.5
        assert gb.remaining == 1.5

    def test_remaining_never_negative(self):
        """Test remaining never goes below zero."""
        gb = GlobalBudget(max_cost_usd=1.0)
        gb.record_cost(5.0)
        assert gb.remaining == 0.0

    def test_check_cost_under_budget(self):
        """Test check_cost passes when under budget."""
        gb = GlobalBudget(max_cost_usd=5.0)
        gb.record_cost(2.0)
        gb.check_cost()  # Should not raise

    def test_check_cost_at_budget_raises(self):
        """Test check_cost raises when at budget."""
        gb = GlobalBudget(max_cost_usd=5.0)
        gb.record_cost(5.0)
        with pytest.raises(BudgetExceededError):
            gb.check_cost()

    def test_check_cost_over_budget_raises(self):
        """Test check_cost raises when over budget."""
        gb = GlobalBudget(max_cost_usd=5.0)
        gb.record_cost(7.0)
        with pytest.raises(BudgetExceededError) as exc_info:
            gb.check_cost()
        assert "Global budget exceeded" in str(exc_info.value)


class TestGlobalBudgetTimeTracking:
    """Test GlobalBudget time tracking."""

    def test_elapsed_seconds(self):
        """Test elapsed_seconds is tracked."""
        gb = GlobalBudget(max_cost_usd=5.0)
        assert gb.elapsed_seconds >= 0.0
        assert gb.elapsed_seconds < 1.0  # Should be very fast

    def test_check_time_no_limit(self):
        """Test check_time is a no-op when max_seconds is None."""
        gb = GlobalBudget(max_cost_usd=5.0)
        gb.check_time()  # Should not raise

    def test_check_time_within_limit(self):
        """Test check_time passes when within limit."""
        gb = GlobalBudget(max_cost_usd=5.0, max_seconds=60.0)
        gb.check_time()  # Should not raise (just started)

    def test_check_time_expired_raises(self):
        """Test check_time raises when time expired."""
        gb = GlobalBudget(max_cost_usd=5.0, max_seconds=0.01)
        # Manually set start_time to the past
        gb._start_time = time.time() - 1.0
        with pytest.raises(TimeoutExceededError) as exc_info:
            gb.check_time()
        assert "Global timeout exceeded" in str(exc_info.value)


class TestGlobalBudgetReset:
    """Test GlobalBudget reset."""

    def test_reset_clears_cost(self):
        """Test reset clears accumulated cost."""
        gb = GlobalBudget(max_cost_usd=5.0)
        gb.record_cost(3.0)
        assert gb.total_spent == 3.0
        gb.reset()
        assert gb.total_spent == 0.0
        assert gb.remaining == 5.0

    def test_reset_resets_timer(self):
        """Test reset resets the start time."""
        gb = GlobalBudget(max_cost_usd=5.0, max_seconds=60.0)
        gb._start_time = time.time() - 100.0  # Pretend 100s elapsed
        gb.reset()
        assert gb.elapsed_seconds < 1.0


class TestGlobalBudgetThreadSafety:
    """Test GlobalBudget thread safety."""

    def test_concurrent_record_cost(self):
        """Test concurrent cost recording is thread-safe."""
        gb = GlobalBudget(max_cost_usd=1000.0)
        num_threads = 10
        cost_per_thread = 1.0
        iterations = 100

        def record_costs():
            for _ in range(iterations):
                gb.record_cost(cost_per_thread)

        threads = [threading.Thread(target=record_costs) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = num_threads * iterations * cost_per_thread
        assert gb.total_spent == expected

    def test_concurrent_check_and_record(self):
        """Test concurrent check and record operations."""
        gb = GlobalBudget(max_cost_usd=1000.0)
        errors = []

        def check_and_record():
            try:
                for _ in range(50):
                    gb.record_cost(0.01)
                    gb.check_cost()
            except BudgetExceededError:
                errors.append("budget_exceeded")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=check_and_record) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No unexpected errors
        unexpected = [e for e in errors if e != "budget_exceeded"]
        assert len(unexpected) == 0


# ============================================================================
# Error Type Tests
# ============================================================================

class TestBudgetErrors:
    """Test budget-related error types."""

    def test_budget_exceeded_error_is_agentcircuit_error(self):
        """Test BudgetExceededError inherits from AgentCircuitError."""
        from agentcircuit.errors import AgentCircuitError
        assert issubclass(BudgetExceededError, AgentCircuitError)

    def test_timeout_exceeded_error_is_agentcircuit_error(self):
        """Test TimeoutExceededError inherits from AgentCircuitError."""
        from agentcircuit.errors import AgentCircuitError
        assert issubclass(TimeoutExceededError, AgentCircuitError)

    def test_budget_exceeded_error_attributes(self):
        """Test BudgetExceededError stores spent and limit."""
        err = BudgetExceededError("test", spent=1.5, limit=1.0)
        assert err.spent == 1.5
        assert err.limit == 1.0
        assert str(err) == "test"

    def test_timeout_exceeded_error_attributes(self):
        """Test TimeoutExceededError stores elapsed and limit."""
        err = TimeoutExceededError("test", elapsed=30.0, limit=10.0)
        assert err.elapsed == 30.0
        assert err.limit == 10.0
        assert str(err) == "test"
