"""
Pytest configuration and shared fixtures for AgentCircuit tests.
"""
import pytest
import tempfile
import os
import shutil
from typing import Generator, Dict, Any

from pydantic import BaseModel


# ============================================================================
# Test Schemas
# ============================================================================

class SimpleOutputSchema(BaseModel):
    """Simple schema for testing validation."""
    message: str
    status: str


class ComplexOutputSchema(BaseModel):
    """Complex schema with nested fields."""
    summary: str
    confidence: float
    metadata: Dict[str, Any]


class StrictOutputSchema(BaseModel):
    """Strict schema with constraints."""
    name: str
    age: int
    email: str


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Create a temporary database path for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, ".agentcircuit", "traces.db")
    yield db_path
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_llm_callable():
    """Mock LLM callable that returns valid JSON."""
    def _mock_llm(prompt: str) -> str:
        # Parse the schema from the prompt and return valid JSON
        if "SimpleOutputSchema" in prompt or "message" in prompt:
            return '{"message": "Fixed output", "status": "repaired"}'
        elif "ComplexOutputSchema" in prompt:
            return '{"summary": "Fixed summary", "confidence": 0.95, "metadata": {}}'
        elif "StrictOutputSchema" in prompt:
            return '{"name": "John Doe", "age": 30, "email": "john@example.com"}'
        return '{"result": "default_fix"}'
    return _mock_llm


@pytest.fixture
def failing_llm_callable():
    """Mock LLM callable that always fails."""
    def _failing_llm(prompt: str) -> str:
        raise Exception("LLM API Error: Connection timeout")
    return _failing_llm


@pytest.fixture
def invalid_json_llm_callable():
    """Mock LLM callable that returns invalid JSON."""
    def _invalid_llm(prompt: str) -> str:
        return "This is not valid JSON at all"
    return _invalid_llm


@pytest.fixture
def markdown_wrapped_llm_callable():
    """Mock LLM callable that returns JSON wrapped in markdown."""
    def _markdown_llm(prompt: str) -> str:
        return '''Here is the fixed output:
```json
{"message": "Fixed from markdown", "status": "repaired"}
```
'''
    return _markdown_llm


@pytest.fixture
def sample_state() -> Dict[str, Any]:
    """Sample input state for testing."""
    return {
        "input": "Test input data",
        "messages": ["Hello", "World"],
        "context": {"key": "value"}
    }


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample config for testing."""
    return {
        "configurable": {
            "thread_id": "test-thread-123"
        },
        "run_id": "test-run-456"
    }


# ============================================================================
# Test Utilities
# ============================================================================

class MockGroqClient:
    """Mock Groq client for testing."""

    def __init__(self, response: str = '{"result": "mocked"}'):
        self.response = response
        self.calls = []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, messages, model):
        self.calls.append({"messages": messages, "model": model})

        class MockChoice:
            class MockMessage:
                content = self.response
            message = MockMessage()

        class MockCompletion:
            choices = [MockChoice()]

        return MockCompletion()
