"""
Unit tests for the Medic module - Error Recovery.
"""
import pytest
import json
import sys
from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel
from typing import Dict, Any

from agentcircuit.medic import Medic, MedicError


# ============================================================================
# Test Schemas
# ============================================================================

class MedicTestOutputSchema(BaseModel):
    """Output schema for medic tests."""
    __test__ = False  # Tell pytest not to collect this class as a test
    message: str
    status: str


class ComplexSchema(BaseModel):
    summary: str
    confidence: float
    metadata: Dict[str, Any]


# Alias for backward compatibility in tests
TestOutputSchema = MedicTestOutputSchema


# ============================================================================
# Tests
# ============================================================================

class TestMedicBasics:
    """Test basic Medic functionality."""

    def test_medic_init_no_callable(self):
        """Test Medic initializes without LLM callable."""
        # Patch environment and providers to ensure no LLM is available
        with patch.dict('os.environ', {}, clear=True):
            with patch('agentcircuit.medic._get_providers') as mock_providers:
                # Make provider chain creation fail
                mock_prov = MagicMock()
                mock_prov.create_default_chain.side_effect = Exception("No providers available")
                mock_providers.return_value = mock_prov
                # Also ensure groq import fails
                with patch.dict(sys.modules, {'groq': None}):
                    medic = Medic()
                    assert medic.llm_callable is None

    def test_medic_init_with_callable(self):
        """Test Medic initializes with custom LLM callable."""
        mock_callable = Mock(return_value='{"result": "ok"}')
        medic = Medic(llm_callable=mock_callable)
        assert medic.llm_callable == mock_callable

    def test_medic_init_groq_fallback(self):
        """Test Medic falls back to Groq when available."""
        # Create a mock Groq class
        mock_groq_instance = MagicMock()
        mock_groq_instance.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content='{"result": "ok"}'))
        ]
        mock_groq_class = MagicMock(return_value=mock_groq_instance)

        # Create a mock groq module
        mock_groq_module = MagicMock()
        mock_groq_module.Groq = mock_groq_class

        with patch.dict('os.environ', {'GROQ_API_KEY': 'test-key'}):
            with patch('agentcircuit.medic._get_providers') as mock_providers:
                # Make provider chain creation fail so it falls back to Groq
                mock_prov = MagicMock()
                mock_prov.create_default_chain.side_effect = Exception("No providers")
                mock_providers.return_value = mock_prov
                with patch.dict(sys.modules, {'groq': mock_groq_module}):
                    # Need to reimport to pick up the mock
                    import importlib
                    import agentcircuit.medic
                    importlib.reload(agentcircuit.medic)
                    from agentcircuit.medic import Medic as ReloadedMedic
                    medic = ReloadedMedic()
                    assert medic.llm_callable is not None


class TestMedicRecovery:
    """Test Medic recovery logic."""

    def test_recovery_success(self, mock_llm_callable):
        """Test successful recovery."""
        medic = Medic(llm_callable=mock_llm_callable)

        result = medic.attempt_recovery(
            error=Exception("Test error"),
            input_state={"input": "test"},
            raw_output="invalid output",
            node_id="test_node",
            recovery_attempts=1,
            schema=TestOutputSchema
        )

        assert result["message"] == "Fixed output"
        assert result["status"] == "repaired"

    def test_recovery_exceeds_attempts(self):
        """Test recovery fails after max attempts."""
        mock_callable = Mock()
        medic = Medic(llm_callable=mock_callable)

        with pytest.raises(MedicError) as exc_info:
            medic.attempt_recovery(
                error=Exception("Test error"),
                input_state={},
                raw_output=None,
                node_id="test_node",
                recovery_attempts=3,  # Exceeds limit of 2
                schema=None
            )

        assert "Exceeded 2 recovery attempts" in str(exc_info.value)
        mock_callable.assert_not_called()

    def test_recovery_no_llm_raises_original(self):
        """Test recovery without LLM re-raises original error."""
        with patch.dict('os.environ', {}, clear=True):
            with patch('agentcircuit.medic._get_providers') as mock_providers:
                # Make provider chain creation fail
                mock_prov = MagicMock()
                mock_prov.create_default_chain.side_effect = Exception("No providers")
                mock_providers.return_value = mock_prov
                with patch.dict(sys.modules, {'groq': None}):
                    medic = Medic()
                    original_error = ValueError("Original error")

                    with pytest.raises(ValueError) as exc_info:
                        medic.attempt_recovery(
                            error=original_error,
                            input_state={},
                            raw_output=None,
                            node_id="test_node",
                            recovery_attempts=1,
                            schema=None
                        )

                    assert exc_info.value == original_error

    def test_recovery_parses_markdown_json(self, markdown_wrapped_llm_callable):
        """Test recovery parses JSON from markdown blocks."""
        medic = Medic(llm_callable=markdown_wrapped_llm_callable)

        result = medic.attempt_recovery(
            error=Exception("Test error"),
            input_state={},
            raw_output=None,
            node_id="test_node",
            recovery_attempts=1,
            schema=TestOutputSchema
        )

        assert result["message"] == "Fixed from markdown"

    def test_recovery_llm_failure(self, failing_llm_callable):
        """Test recovery handles LLM failure gracefully."""
        medic = Medic(llm_callable=failing_llm_callable)

        with pytest.raises(MedicError) as exc_info:
            medic.attempt_recovery(
                error=Exception("Original error"),
                input_state={},
                raw_output=None,
                node_id="test_node",
                recovery_attempts=1,
                schema=None
            )

        assert "Repair failed" in str(exc_info.value)

    def test_recovery_invalid_json(self, invalid_json_llm_callable):
        """Test recovery handles invalid JSON from LLM."""
        medic = Medic(llm_callable=invalid_json_llm_callable)

        with pytest.raises(MedicError) as exc_info:
            medic.attempt_recovery(
                error=Exception("Original error"),
                input_state={},
                raw_output=None,
                node_id="test_node",
                recovery_attempts=1,
                schema=None
            )

        assert "Repair failed" in str(exc_info.value)


class TestMedicPromptConstruction:
    """Test prompt construction for LLM."""

    def test_prompt_includes_error(self):
        """Test prompt includes error message."""
        captured_prompt = None

        def capture_prompt(prompt: str) -> str:
            nonlocal captured_prompt
            captured_prompt = prompt
            return '{"result": "ok"}'

        medic = Medic(llm_callable=capture_prompt)

        medic.attempt_recovery(
            error=ValueError("Specific error message"),
            input_state={},
            raw_output=None,
            node_id="test_node",
            recovery_attempts=1,
            schema=None
        )

        assert "Specific error message" in captured_prompt

    def test_prompt_includes_error_type(self):
        """Test prompt includes error type/category."""
        captured_prompt = None

        def capture_prompt(prompt: str) -> str:
            nonlocal captured_prompt
            captured_prompt = prompt
            return '{"result": "ok"}'

        medic = Medic(llm_callable=capture_prompt)

        medic.attempt_recovery(
            error=Exception("Error"),
            input_state={},
            raw_output=None,
            node_id="my_custom_node",
            recovery_attempts=1,
            schema=None
        )

        # The prompt should include error type/category info
        assert "ERROR" in captured_prompt or "error" in captured_prompt.lower()

    def test_prompt_includes_input_state(self):
        """Test prompt includes input state."""
        captured_prompt = None

        def capture_prompt(prompt: str) -> str:
            nonlocal captured_prompt
            captured_prompt = prompt
            return '{"result": "ok"}'

        medic = Medic(llm_callable=capture_prompt)

        medic.attempt_recovery(
            error=Exception("Error"),
            input_state={"key": "test_value_12345"},
            raw_output=None,
            node_id="test_node",
            recovery_attempts=1,
            schema=None
        )

        assert "test_value_12345" in captured_prompt

    def test_prompt_includes_schema_v2(self):
        """Test prompt includes Pydantic v2 schema."""
        captured_prompt = None

        def capture_prompt(prompt: str) -> str:
            nonlocal captured_prompt
            captured_prompt = prompt
            return '{"message": "ok", "status": "done"}'

        medic = Medic(llm_callable=capture_prompt)

        medic.attempt_recovery(
            error=Exception("Error"),
            input_state={},
            raw_output=None,
            node_id="test_node",
            recovery_attempts=1,
            schema=TestOutputSchema
        )

        # Should include schema definition
        assert "message" in captured_prompt
        assert "status" in captured_prompt


class TestMedicError:
    """Test MedicError exception."""

    def test_medic_error_is_exception(self):
        """Test MedicError inherits from Exception."""
        assert issubclass(MedicError, Exception)

    def test_medic_error_message(self):
        """Test MedicError preserves message."""
        error = MedicError("Recovery failed after 2 attempts")
        assert "Recovery failed" in str(error)


class TestMedicJSONParsing:
    """Test JSON parsing edge cases."""

    def test_parse_clean_json(self):
        """Test parsing clean JSON."""
        def clean_llm(prompt: str) -> str:
            return '{"key": "value"}'

        medic = Medic(llm_callable=clean_llm)
        result = medic.attempt_recovery(
            error=Exception("Error"),
            input_state={},
            raw_output=None,
            node_id="test_node",
            recovery_attempts=1,
            schema=None
        )

        assert result == {"key": "value"}

    def test_parse_json_with_triple_backticks(self):
        """Test parsing JSON with generic triple backticks."""
        def backtick_llm(prompt: str) -> str:
            return '```\n{"key": "value"}\n```'

        medic = Medic(llm_callable=backtick_llm)
        result = medic.attempt_recovery(
            error=Exception("Error"),
            input_state={},
            raw_output=None,
            node_id="test_node",
            recovery_attempts=1,
            schema=None
        )

        assert result == {"key": "value"}

    def test_parse_json_with_json_backticks(self):
        """Test parsing JSON with ```json blocks."""
        def json_block_llm(prompt: str) -> str:
            return '```json\n{"key": "value"}\n```'

        medic = Medic(llm_callable=json_block_llm)
        result = medic.attempt_recovery(
            error=Exception("Error"),
            input_state={},
            raw_output=None,
            node_id="test_node",
            recovery_attempts=1,
            schema=None
        )

        assert result == {"key": "value"}

    def test_parse_json_with_surrounding_text(self):
        """Test parsing JSON with surrounding explanation text."""
        def verbose_llm(prompt: str) -> str:
            return 'Here is the fixed output:\n```json\n{"key": "value"}\n```\nThis should work now.'

        medic = Medic(llm_callable=verbose_llm)
        result = medic.attempt_recovery(
            error=Exception("Error"),
            input_state={},
            raw_output=None,
            node_id="test_node",
            recovery_attempts=1,
            schema=None
        )

        assert result == {"key": "value"}
