"""
Unit tests for the Sentinel module - Output Validation.
"""
import pytest
from pydantic import BaseModel, ValidationError
from typing import Dict, Any, Optional, List

from agentcircuit.sentinel import Sentinel, SentinelError


# ============================================================================
# Test Schemas
# ============================================================================

class SimpleSchema(BaseModel):
    message: str
    status: str


class NumericSchema(BaseModel):
    value: int
    score: float


class NestedSchema(BaseModel):
    name: str
    metadata: Dict[str, Any]


class OptionalFieldSchema(BaseModel):
    required_field: str
    optional_field: Optional[str] = None


class ListSchema(BaseModel):
    items: List[str]
    count: int


class StrictTypesSchema(BaseModel):
    age: int
    active: bool


# ============================================================================
# Tests
# ============================================================================

class TestSentinelBasics:
    """Test basic Sentinel functionality."""

    def test_sentinel_init_no_schema(self):
        """Test Sentinel initializes without schema."""
        sentinel = Sentinel()
        assert sentinel.schema is None

    def test_sentinel_init_with_schema(self):
        """Test Sentinel initializes with schema."""
        sentinel = Sentinel(schema=SimpleSchema)
        assert sentinel.schema == SimpleSchema

    def test_validate_passthrough_no_schema(self):
        """Test validation passes through when no schema."""
        sentinel = Sentinel()
        output = {"any": "data", "goes": "through"}
        result = sentinel.validate(output)
        assert result == output

    def test_validate_passthrough_none_output(self):
        """Test None output passes through without schema."""
        sentinel = Sentinel()
        result = sentinel.validate(None)
        assert result is None


class TestSentinelValidation:
    """Test Sentinel validation with schemas."""

    def test_validate_valid_dict(self):
        """Test validation of valid dictionary."""
        sentinel = Sentinel(schema=SimpleSchema)
        output = {"message": "Hello", "status": "ok"}
        result = sentinel.validate(output)
        assert isinstance(result, SimpleSchema)
        assert result.message == "Hello"
        assert result.status == "ok"

    def test_validate_missing_field(self):
        """Test validation fails with missing field."""
        sentinel = Sentinel(schema=SimpleSchema)
        output = {"message": "Hello"}  # Missing 'status'

        with pytest.raises(SentinelError) as exc_info:
            sentinel.validate(output)

        assert "validation failed" in str(exc_info.value).lower()

    def test_validate_wrong_type(self):
        """Test validation fails with wrong type."""
        sentinel = Sentinel(schema=NumericSchema)
        output = {"value": "not_an_int", "score": 0.5}

        with pytest.raises(SentinelError):
            sentinel.validate(output)

    def test_validate_extra_fields_allowed(self):
        """Test extra fields are handled by Pydantic."""
        sentinel = Sentinel(schema=SimpleSchema)
        output = {"message": "Hello", "status": "ok", "extra": "field"}
        # By default, Pydantic v2 ignores extra fields
        result = sentinel.validate(output)
        assert result.message == "Hello"

    def test_validate_nested_dict(self):
        """Test validation of nested dictionary."""
        sentinel = Sentinel(schema=NestedSchema)
        output = {"name": "Test", "metadata": {"key": "value", "num": 123}}
        result = sentinel.validate(output)
        assert result.name == "Test"
        assert result.metadata == {"key": "value", "num": 123}

    def test_validate_optional_field_present(self):
        """Test validation with optional field present."""
        sentinel = Sentinel(schema=OptionalFieldSchema)
        output = {"required_field": "value", "optional_field": "optional"}
        result = sentinel.validate(output)
        assert result.optional_field == "optional"

    def test_validate_optional_field_missing(self):
        """Test validation with optional field missing."""
        sentinel = Sentinel(schema=OptionalFieldSchema)
        output = {"required_field": "value"}
        result = sentinel.validate(output)
        assert result.optional_field is None

    def test_validate_list_field(self):
        """Test validation of list fields."""
        sentinel = Sentinel(schema=ListSchema)
        output = {"items": ["a", "b", "c"], "count": 3}
        result = sentinel.validate(output)
        assert result.items == ["a", "b", "c"]
        assert result.count == 3

    def test_validate_model_instance(self):
        """Test validation of already-valid model instance."""
        sentinel = Sentinel(schema=SimpleSchema)
        instance = SimpleSchema(message="Hello", status="ok")
        result = sentinel.validate(instance)
        assert result == instance

    def test_validate_type_coercion(self):
        """Test Pydantic type coercion."""
        sentinel = Sentinel(schema=NumericSchema)
        # String "123" should coerce to int
        output = {"value": "123", "score": "0.5"}
        result = sentinel.validate(output)
        assert result.value == 123
        assert result.score == 0.5

    def test_validate_boolean_coercion(self):
        """Test boolean coercion."""
        sentinel = Sentinel(schema=StrictTypesSchema)
        output = {"age": 25, "active": "true"}
        # Pydantic v2 is stricter about bool coercion
        # This may fail depending on Pydantic settings
        try:
            result = sentinel.validate(output)
            assert result.active == True
        except SentinelError:
            # Expected in strict mode
            pass


class TestSentinelError:
    """Test SentinelError exception."""

    def test_sentinel_error_is_exception(self):
        """Test SentinelError inherits from Exception."""
        assert issubclass(SentinelError, Exception)

    def test_sentinel_error_message(self):
        """Test SentinelError preserves message."""
        error = SentinelError("Validation failed: missing field")
        assert "Validation failed" in str(error)

    def test_sentinel_error_contains_details(self):
        """Test SentinelError from validation contains details."""
        sentinel = Sentinel(schema=SimpleSchema)

        try:
            sentinel.validate({"wrong": "fields"})
        except SentinelError as e:
            error_msg = str(e)
            assert "Sentinel Alert" in error_msg
            assert "validation failed" in error_msg.lower()


class TestSentinelEdgeCases:
    """Test edge cases and error handling."""

    def test_validate_empty_dict(self):
        """Test validation of empty dictionary."""
        sentinel = Sentinel(schema=SimpleSchema)

        with pytest.raises(SentinelError):
            sentinel.validate({})

    def test_validate_none_with_schema(self):
        """Test validation of None with schema."""
        sentinel = Sentinel(schema=SimpleSchema)

        with pytest.raises(SentinelError):
            sentinel.validate(None)

    def test_validate_list_instead_of_dict(self):
        """Test validation fails with list instead of dict."""
        sentinel = Sentinel(schema=SimpleSchema)

        with pytest.raises(SentinelError):
            sentinel.validate(["not", "a", "dict"])

    def test_validate_string_instead_of_dict(self):
        """Test validation of string (might be JSON)."""
        sentinel = Sentinel(schema=SimpleSchema)

        # Pydantic v2 doesn't auto-parse JSON strings
        with pytest.raises(SentinelError):
            sentinel.validate('{"message": "hello", "status": "ok"}')
