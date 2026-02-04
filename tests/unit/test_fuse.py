"""
Unit tests for the Fuse module - Loop Detection.
"""
import pytest
from agentcircuit.fuse import Fuse, LoopError


class TestFuseBasics:
    """Test basic Fuse functionality."""

    def test_fuse_init_default_limit(self):
        """Test Fuse initializes with default limit of 3."""
        fuse = Fuse()
        assert fuse.limit == 3

    def test_fuse_init_custom_limit(self):
        """Test Fuse initializes with custom limit."""
        fuse = Fuse(limit=5)
        assert fuse.limit == 5

    def test_fuse_init_limit_one(self):
        """Test Fuse with limit of 1 (immediate trip on repeat)."""
        fuse = Fuse(limit=1)
        assert fuse.limit == 1


class TestFuseHashing:
    """Test state hashing functionality."""

    def test_hash_simple_dict(self):
        """Test hashing a simple dictionary."""
        fuse = Fuse()
        state = {"key": "value"}
        hash1 = fuse._hash_state(state)
        hash2 = fuse._hash_state(state)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_hash_different_dicts(self):
        """Test different dicts produce different hashes."""
        fuse = Fuse()
        state1 = {"key": "value1"}
        state2 = {"key": "value2"}
        assert fuse._hash_state(state1) != fuse._hash_state(state2)

    def test_hash_key_order_invariant(self):
        """Test hash is same regardless of key order."""
        fuse = Fuse()
        state1 = {"a": 1, "b": 2}
        state2 = {"b": 2, "a": 1}
        assert fuse._hash_state(state1) == fuse._hash_state(state2)

    def test_hash_nested_dict(self):
        """Test hashing nested dictionaries."""
        fuse = Fuse()
        state = {"outer": {"inner": {"deep": "value"}}}
        hash1 = fuse._hash_state(state)
        hash2 = fuse._hash_state(state)
        assert hash1 == hash2

    def test_hash_with_list(self):
        """Test hashing state with lists."""
        fuse = Fuse()
        state = {"items": [1, 2, 3], "name": "test"}
        hash1 = fuse._hash_state(state)
        hash2 = fuse._hash_state(state)
        assert hash1 == hash2

    def test_hash_non_serializable_fallback(self):
        """Test fallback for non-serializable objects."""
        fuse = Fuse()

        class NonSerializable:
            def __init__(self):
                self.value = "test"

        state = {"obj": NonSerializable()}
        # Should not raise, uses str() fallback
        hash_result = fuse._hash_state(state)
        assert hash_result is not None


class TestFuseLoopDetection:
    """Test loop detection logic."""

    def test_no_loop_empty_history(self):
        """Test no loop with empty history."""
        fuse = Fuse(limit=3)
        state = {"input": "test"}
        # Should not raise
        fuse.check(history=[], current_state=state)

    def test_no_loop_different_states(self):
        """Test no loop when states are different."""
        fuse = Fuse(limit=3)
        history = [
            fuse._hash_state({"input": "state1"}),
            fuse._hash_state({"input": "state2"}),
            fuse._hash_state({"input": "state3"}),
        ]
        # Should not raise
        fuse.check(history=history, current_state={"input": "state4"})

    def test_loop_detected_at_limit(self):
        """Test loop is detected when limit is reached."""
        fuse = Fuse(limit=3)
        repeated_state = {"input": "repeated"}
        history = [fuse._hash_state(repeated_state)] * 3

        with pytest.raises(LoopError) as exc_info:
            fuse.check(history=history, current_state=repeated_state)

        assert "Loop detected" in str(exc_info.value)
        assert "3 times" in str(exc_info.value)

    def test_no_loop_below_limit(self):
        """Test no loop when count is below limit."""
        fuse = Fuse(limit=3)
        repeated_state = {"input": "repeated"}
        history = [fuse._hash_state(repeated_state)] * 2

        # Should not raise - only 2 repeats, limit is 3
        fuse.check(history=history, current_state=repeated_state)

    def test_loop_with_mixed_history(self):
        """Test loop detection with mixed history."""
        fuse = Fuse(limit=2)
        looping_state = {"input": "loop"}
        history = [
            fuse._hash_state({"input": "other1"}),
            fuse._hash_state(looping_state),
            fuse._hash_state({"input": "other2"}),
            fuse._hash_state(looping_state),
        ]

        with pytest.raises(LoopError):
            fuse.check(history=history, current_state=looping_state)

    def test_loop_limit_one(self):
        """Test immediate loop detection with limit=1."""
        fuse = Fuse(limit=1)
        state = {"input": "test"}
        history = [fuse._hash_state(state)]

        with pytest.raises(LoopError):
            fuse.check(history=history, current_state=state)


class TestLoopError:
    """Test LoopError exception."""

    def test_loop_error_is_exception(self):
        """Test LoopError inherits from Exception."""
        assert issubclass(LoopError, Exception)

    def test_loop_error_message(self):
        """Test LoopError preserves message."""
        error = LoopError("Test error message")
        assert str(error) == "Test error message"
