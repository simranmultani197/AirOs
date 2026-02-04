from typing import Any, Type, Optional
from pydantic import BaseModel, ValidationError

class Sentinel:
    """
    The Sentinel: Validates node output against a Pydantic schema.
    """
    def __init__(self, schema: Optional[Type[BaseModel]] = None):
        self.schema = schema

    def validate(self, output: Any) -> Any:
        """
        Validates output. If schema is present, parses it.
        Returns the validated object (or dict).
        Raises ValidationError if invalid.
        """
        if not self.schema:
            return output
        
        try:
            if isinstance(output, dict):
                return self.schema.model_validate(output)
            elif isinstance(output, self.schema):
                return output
            else:
                # Try to parse from string if it's a string (e.g. JSON string)
                # But typically LangGraph nodes return dicts or objects.
                return self.schema.model_validate(output)
        except ValidationError as e:
            raise SentinelError(f"Sentinel Alert: Output validation failed: {e}") from e

class SentinelError(Exception):
    pass
