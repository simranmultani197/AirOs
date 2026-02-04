"""
Cost-saving circuit breakers for AgentCircuit.

Provides:
- BudgetFuse: Trips when cumulative dollar spend exceeds a threshold
- TimeoutFuse: Trips when execution time exceeds a limit
- GlobalBudget: Thread-safe shared budget across multiple nodes/runs
"""
import time
import threading
from typing import Optional

from .errors import BudgetExceededError, TimeoutExceededError


class BudgetFuse:
    """
    Dollar-based circuit breaker.

    Trips when the cumulative cost for a run exceeds max_cost_usd.
    Queries the storage layer to get the current run cost.

    Usage:
        @reliable(max_cost_usd=1.0)
        def my_node(state):
            return expensive_llm_call(state)
    """

    def __init__(self, max_cost_usd: float):
        if max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be positive")
        self.max_cost_usd = max_cost_usd

    def check(self, current_cost: float) -> None:
        """
        Check if the current cumulative cost exceeds the budget.

        Args:
            current_cost: The current cumulative cost for this run

        Raises:
            BudgetExceededError: If the budget is exceeded
        """
        if current_cost >= self.max_cost_usd:
            raise BudgetExceededError(
                f"Budget exceeded: ${current_cost:.4f} >= ${self.max_cost_usd:.4f} limit",
                spent=current_cost,
                limit=self.max_cost_usd,
            )


class TimeoutFuse:
    """
    Time-based circuit breaker.

    Trips when elapsed execution time exceeds max_seconds.

    Usage:
        @reliable(max_seconds=30)
        def my_node(state):
            return slow_operation(state)
    """

    def __init__(self, max_seconds: float):
        if max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        self.max_seconds = max_seconds

    def check(self, start_time: float) -> None:
        """
        Check if elapsed time exceeds the timeout.

        Args:
            start_time: The time.time() when execution started

        Raises:
            TimeoutExceededError: If the timeout is exceeded
        """
        elapsed = time.time() - start_time
        if elapsed >= self.max_seconds:
            raise TimeoutExceededError(
                f"Timeout exceeded: {elapsed:.1f}s >= {self.max_seconds:.1f}s limit",
                elapsed=elapsed,
                limit=self.max_seconds,
            )


class GlobalBudget:
    """
    Thread-safe shared budget across multiple nodes/runs.

    Share a single GlobalBudget instance across multiple @reliable-decorated
    functions to enforce a total cost and/or time limit for an entire
    agent graph execution.

    Usage:
        from agentcircuit import reliable, GlobalBudget

        budget = GlobalBudget(max_cost_usd=5.0, max_seconds=120)

        @reliable(budget=budget)
        def node_a(state):
            ...

        @reliable(budget=budget)
        def node_b(state):
            ...

        # After execution:
        print(f"Total spent: ${budget.total_spent:.4f}")
        print(f"Remaining: ${budget.remaining:.4f}")
    """

    def __init__(
        self,
        max_cost_usd: float,
        max_seconds: Optional[float] = None,
    ):
        if max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be positive")
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError("max_seconds must be positive")

        self.max_cost_usd = max_cost_usd
        self.max_seconds = max_seconds
        self._total_spent = 0.0
        self._start_time = time.time()
        self._lock = threading.Lock()

    def check_cost(self) -> None:
        """
        Check if the total spent exceeds the budget.

        Raises:
            BudgetExceededError: If the budget is exceeded
        """
        with self._lock:
            if self._total_spent >= self.max_cost_usd:
                raise BudgetExceededError(
                    f"Global budget exceeded: ${self._total_spent:.4f} >= ${self.max_cost_usd:.4f} limit",
                    spent=self._total_spent,
                    limit=self.max_cost_usd,
                )

    def check_time(self) -> None:
        """
        Check if the elapsed time exceeds the time limit.

        Raises:
            TimeoutExceededError: If the time limit is exceeded
        """
        if self.max_seconds is None:
            return

        elapsed = time.time() - self._start_time
        if elapsed >= self.max_seconds:
            raise TimeoutExceededError(
                f"Global timeout exceeded: {elapsed:.1f}s >= {self.max_seconds:.1f}s limit",
                elapsed=elapsed,
                limit=self.max_seconds,
            )

    def record_cost(self, cost: float) -> None:
        """
        Record a cost incurred by a node execution.

        Args:
            cost: The cost to add to the running total
        """
        with self._lock:
            self._total_spent += cost

    @property
    def total_spent(self) -> float:
        """Get the total amount spent so far."""
        with self._lock:
            return self._total_spent

    @property
    def remaining(self) -> float:
        """Get the remaining budget."""
        with self._lock:
            return max(0.0, self.max_cost_usd - self._total_spent)

    @property
    def elapsed_seconds(self) -> float:
        """Get the elapsed time since the budget was created."""
        return time.time() - self._start_time

    def reset(self) -> None:
        """Reset the budget (useful for restarting a graph run)."""
        with self._lock:
            self._total_spent = 0.0
            self._start_time = time.time()
