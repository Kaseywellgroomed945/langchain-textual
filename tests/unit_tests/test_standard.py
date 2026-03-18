"""Standard LangChain unit tests for Tonic Textual tools."""

from typing import Any

from langchain_tests.unit_tests import ToolsUnitTests

from langchain_textual import (
    TonicTextualPiiTypes,
    TonicTextualRedactFile,
    TonicTextualRedactHtml,
    TonicTextualRedactJson,
    TonicTextualRedactText,
)


class TestTonicTextualRedactTextUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[TonicTextualRedactText]:
        return TonicTextualRedactText

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"tonic_textual_api_key": "fake-api-key"}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "My name is John Smith and I live in Atlanta, GA."}

    @property
    def init_from_env_params(
        self,
    ) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return (
            {"TONIC_TEXTUAL_API_KEY": "test-api-key"},
            {},
            {"tonic_textual_api_key": "test-api-key"},
        )


class TestTonicTextualRedactJsonUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[TonicTextualRedactJson]:
        return TonicTextualRedactJson

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"tonic_textual_api_key": "fake-api-key"}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"json_str": '{"name": "John Smith", "email": "john@example.com"}'}

    @property
    def init_from_env_params(
        self,
    ) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return (
            {"TONIC_TEXTUAL_API_KEY": "test-api-key"},
            {},
            {"tonic_textual_api_key": "test-api-key"},
        )


class TestTonicTextualRedactHtmlUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[TonicTextualRedactHtml]:
        return TonicTextualRedactHtml

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"tonic_textual_api_key": "fake-api-key"}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"html_str": "<p>Contact John Smith at john@example.com</p>"}

    @property
    def init_from_env_params(
        self,
    ) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return (
            {"TONIC_TEXTUAL_API_KEY": "test-api-key"},
            {},
            {"tonic_textual_api_key": "test-api-key"},
        )


class TestTonicTextualRedactFileUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[TonicTextualRedactFile]:
        return TonicTextualRedactFile

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"tonic_textual_api_key": "fake-api-key"}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"file_path": "/tmp/test.pdf"}

    @property
    def init_from_env_params(
        self,
    ) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return (
            {"TONIC_TEXTUAL_API_KEY": "test-api-key"},
            {},
            {"tonic_textual_api_key": "test-api-key"},
        )


class TestTonicTextualPiiTypesUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[TonicTextualPiiTypes]:
        return TonicTextualPiiTypes

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": ""}
