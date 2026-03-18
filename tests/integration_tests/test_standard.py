"""Standard LangChain integration tests for Tonic Textual tools.

These tests require TONIC_TEXTUAL_API_KEY to be set in the environment.
"""

from typing import Any

from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_textual import (
    TonicTextualRedactHtml,
    TonicTextualRedactJson,
    TonicTextualRedactText,
)


class TestTonicTextualRedactTextIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[TonicTextualRedactText]:
        return TonicTextualRedactText

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "My name is John Smith and I live in Atlanta, GA."}


class TestTonicTextualRedactJsonIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[TonicTextualRedactJson]:
        return TonicTextualRedactJson

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "json_str": '{"name": "John Smith", "email": "john@example.com"}'
        }


class TestTonicTextualRedactHtmlIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[TonicTextualRedactHtml]:
        return TonicTextualRedactHtml

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "html_str": "<p>Contact John Smith at john@example.com</p>"
        }
