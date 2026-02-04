import json
import hashlib
from typing import Any, List, Dict

class Fuse:
    """
    The Fuse: Detects execution loops by tracking state hashes.
    Trips if identical state is seen 'limit' times (default 3).
    """
    def __init__(self, limit: int = 3):
        self.limit = limit
        # In a real persistence scenario, this might need to be loaded from DB
        # For now, we assume this is instantiated per graph run or we pass history in.
        # Check if we can store it in the state? 
        # Or we rely on the storage layer to query past states.
        pass

    def check(self, history: List[str], current_state: Any) -> None:
        """
        Checks if current_state has appeared in history more than limit times.
        Raises LoopError if tripped.
        """
        current_hash = self._hash_state(current_state)
        count = history.count(current_hash)
        if count >= self.limit:
            raise LoopError(f"Fuse Tripped: Loop detected. State repeated {count} times.")

    def _hash_state(self, state: Any) -> str:
        """Stable hash of the state."""
        try:
            # Sort keys for consistent JSON
            serialized = json.dumps(state, sort_keys=True, default=str)
            return hashlib.sha256(serialized.encode()).hexdigest()
        except Exception:
            # Fallback for non-serializable
            return str(hash(str(state)))

class LoopError(Exception):
    pass
