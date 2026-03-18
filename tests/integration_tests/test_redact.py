"""Integration tests for Tonic Textual redaction tools.

These tests require TONIC_TEXTUAL_API_KEY to be set in the environment.
"""

import pytest

from langchain_textual import (
    TonicTextualRedactHtml,
    TonicTextualRedactJson,
    TonicTextualRedactText,
)


@pytest.fixture
def tool() -> TonicTextualRedactText:
    """Create a TonicTextualRedactText tool instance."""
    return TonicTextualRedactText()


def test_redact_basic(tool: TonicTextualRedactText) -> None:
    """Test basic PII redaction."""
    result = tool.invoke("My name is John Smith and I live in Atlanta, GA.")
    assert isinstance(result, str)
    assert "John Smith" not in result
    assert "Atlanta" not in result


def test_redact_no_pii(tool: TonicTextualRedactText) -> None:
    """Test text with no PII passes through."""
    result = tool.invoke("The weather is nice today.")
    assert isinstance(result, str)


def test_redact_with_synthesis() -> None:
    """Test redaction with synthesis mode."""
    tool = TonicTextualRedactText(generator_default="Synthesis")
    result = tool.invoke("Contact Jane Doe at jane.doe@example.com.")
    assert isinstance(result, str)
    assert "jane.doe@example.com" not in result


def test_redact_json_basic() -> None:
    """Test basic JSON PII redaction."""
    tool = TonicTextualRedactJson()
    result = tool.invoke('{"name": "John Smith", "email": "john@example.com"}')
    assert isinstance(result, str)
    assert "John Smith" not in result
    assert "john@example.com" not in result


def test_redact_html_basic() -> None:
    """Test basic HTML PII redaction."""
    tool = TonicTextualRedactHtml()
    result = tool.invoke("<p>Contact John Smith at john@example.com</p>")
    assert isinstance(result, str)
    assert "John Smith" not in result
    assert "john@example.com" not in result
