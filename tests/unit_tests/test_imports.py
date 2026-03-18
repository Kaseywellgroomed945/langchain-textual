"""Test that all expected imports are available."""

from langchain_textual import __all__

EXPECTED_ALL = [
    "TonicTextualPiiTypes",
    "TonicTextualRedactFile",
    "TonicTextualRedactHtml",
    "TonicTextualRedactJson",
    "TonicTextualRedactText",
]


def test_all_imports() -> None:
    """Test that __all__ matches expected exports."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)
