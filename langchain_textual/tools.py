"""Tools for Tonic Textual PII redaction."""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, SecretStr, model_validator
from tonic_textual.enums.pii_type import PiiType  # type: ignore[import-untyped]
from tonic_textual.redact_api import TextualNer  # type: ignore[import-untyped]

from langchain_textual._utilities import initialize_client

_TEXT_TOOL_REDIRECT = (
    "Use tonic_textual_redact for .txt files (read contents first)."
)
_JSON_TOOL_REDIRECT = (
    "Use tonic_textual_redact_json for .json files (read contents first)."
)
_HTML_TOOL_REDIRECT = (
    "Use tonic_textual_redact_html for .html/.htm files (read contents first)."
)
_FILE_TOOL_REDIRECT = (
    "Use tonic_textual_redact_file for binary files (PDF, JPG, PNG, CSV, TSV)."
)
_SUPPORTED_FILE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf", ".csv", ".tsv"}


class _RedactTextInput(BaseModel):
    """Input for the plain text redaction tool."""

    text: str = Field(
        description=(
            "Plain text that may contain PII. "
            "For .txt files, read the file first and pass the contents here."
        )
    )


class _RedactJsonInput(BaseModel):
    """Input for the JSON redaction tool."""

    json_str: str = Field(
        description=(
            "A JSON string that may contain PII values. "
            "For .json files, read the file first and pass the contents here."
        )
    )


class _RedactHtmlInput(BaseModel):
    """Input for the HTML redaction tool."""

    html_str: str = Field(
        description=(
            "An HTML string that may contain PII. "
            "For .html or .htm files, read the file first and pass the "
            "contents here."
        )
    )


class _RedactFileInput(BaseModel):
    """Input for the file redaction tool."""

    file_path: str = Field(
        description=(
            "Absolute path to the file to redact. "
            "Supported types: JPG, PNG, PDF, CSV, TSV. "
            "Do NOT use for .txt, .json, .html, or .htm files — use the "
            "dedicated text, JSON, or HTML redaction tools instead."
        )
    )
    output_path: str | None = Field(
        default=None,
        description=(
            "Path to write the redacted file. "
            "Defaults to <original_name>_redacted.<ext> in the same directory."
        ),
    )


class _PiiTypesInput(BaseModel):
    """Input for the PII types listing tool."""

    query: str = Field(
        default="",
        description="Unused. No input is required — pass an empty string.",
    )


class _BaseTonicTextual(BaseTool):
    """Shared client setup for all Tonic Textual tools."""

    client: TextualNer = Field(default=None)  # type: ignore[assignment]
    tonic_textual_api_key: SecretStr = Field(default=SecretStr(""))
    tonic_textual_base_url: str | None = None
    generator_default: Literal["Off", "Redaction", "Synthesis"] | None = None
    generator_config: dict[str, Literal["Off", "Redaction", "Synthesis"]] = Field(
        default_factory=dict
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment and initialize the Textual client."""
        return initialize_client(values)

    def _build_kwargs(self) -> dict[str, Any]:
        """Build shared keyword arguments for the Textual API call."""
        kwargs: dict[str, Any] = {}
        if self.generator_default is not None:
            kwargs["generator_default"] = self.generator_default
        if self.generator_config:
            kwargs["generator_config"] = self.generator_config
        return kwargs


class TonicTextualRedactText(_BaseTonicTextual):
    """Redact PII from plain text using Tonic Textual.

    Use this tool for raw text strings or when reading the contents of .txt
    files. For .json files use ``TonicTextualRedactJson``, for .html/.htm files
    use ``TonicTextualRedactHtml``, and for binary files (PDF, images, CSV, TSV)
    use ``TonicTextualRedactFile``.

    Setup:
        Install ``langchain-textual`` and set environment variable
        ``TONIC_TEXTUAL_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-textual
            export TONIC_TEXTUAL_API_KEY="your-api-key"

    Instantiation:
        .. code-block:: python

            from langchain_textual import TonicTextualRedactText

            tool = TonicTextualRedactText()

    Invocation:
        .. code-block:: python

            tool.invoke("My name is John and I live in Atlanta, GA.")
    """

    name: str = "tonic_textual_redact"
    description: str = (
        "Redacts personally identifiable information (PII) from plain text. "
        "Input should be plain text that may contain PII such as names, "
        "addresses, phone numbers, emails, or other sensitive data. "
        "Output is the text with PII entities redacted or replaced. "
        "For .txt files, read the file contents and pass the text to this tool. "
        "Do NOT use this tool for JSON, HTML, or binary files."
    )
    args_schema: type[BaseModel] = _RedactTextInput

    def _run(
        self,
        text: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Redact PII from the provided text.

        Args:
            text: The plain text to redact PII from.
            run_manager: The run manager for callbacks.

        Returns:
            The redacted text with PII entities replaced.
        """
        if not text or not text.strip():
            return "Error: empty input. Provide plain text containing PII."
        try:
            json.loads(text)
            return (
                "Error: input looks like JSON, not plain text. "
                + _JSON_TOOL_REDIRECT
            )
        except (json.JSONDecodeError, TypeError):
            pass
        if text.strip().startswith(("<html", "<!doctype", "<head", "<body")):
            return (
                "Error: input looks like HTML, not plain text. "
                + _HTML_TOOL_REDIRECT
            )
        try:
            response = self.client.redact(text, **self._build_kwargs())
            return response.redacted_text
        except Exception as e:
            return f"Error redacting text: {e}"


class TonicTextualRedactJson(_BaseTonicTextual):
    """Redact PII from JSON data using Tonic Textual.

    Use this tool for raw JSON strings or when reading the contents of .json
    files. Read the file contents and pass the JSON string to this tool.

    Setup:
        Install ``langchain-textual`` and set environment variable
        ``TONIC_TEXTUAL_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-textual
            export TONIC_TEXTUAL_API_KEY="your-api-key"

    Instantiation:
        .. code-block:: python

            from langchain_textual import TonicTextualRedactJson

            tool = TonicTextualRedactJson()

    Invocation:
        .. code-block:: python

            tool.invoke('{"name": "John Smith", "email": "john@example.com"}')
    """

    name: str = "tonic_textual_redact_json"
    description: str = (
        "Redacts personally identifiable information (PII) from JSON data. "
        "Input should be a JSON string that may contain PII values such as "
        "names, addresses, phone numbers, emails, or other sensitive data. "
        "Output is the JSON with PII values redacted or replaced. "
        "For .json files, read the file contents and pass the JSON string to "
        "this tool. Do NOT use this tool for plain text, HTML, or binary files."
    )
    args_schema: type[BaseModel] = _RedactJsonInput

    def _run(
        self,
        json_str: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Redact PII from the provided JSON data.

        Args:
            json_str: A JSON string to redact PII from.
            run_manager: The run manager for callbacks.

        Returns:
            The redacted JSON string with PII values replaced.
        """
        if not json_str or not json_str.strip():
            return "Error: empty input. Provide a JSON string containing PII."
        try:
            json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return (
                "Error: input is not valid JSON. "
                "Provide a valid JSON string. "
                "If this is plain text, " + _TEXT_TOOL_REDIRECT
            )
        try:
            response = self.client.redact_json(json_str, **self._build_kwargs())
            return response.redacted_text
        except Exception as e:
            return f"Error redacting JSON: {e}"


class TonicTextualRedactHtml(_BaseTonicTextual):
    """Redact PII from HTML content using Tonic Textual.

    Use this tool for raw HTML strings or when reading the contents of .html
    or .htm files. Read the file contents and pass the HTML string to this tool.

    Setup:
        Install ``langchain-textual`` and set environment variable
        ``TONIC_TEXTUAL_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-textual
            export TONIC_TEXTUAL_API_KEY="your-api-key"

    Instantiation:
        .. code-block:: python

            from langchain_textual import TonicTextualRedactHtml

            tool = TonicTextualRedactHtml()

    Invocation:
        .. code-block:: python

            tool.invoke("<p>Contact John Smith at john@example.com</p>")
    """

    name: str = "tonic_textual_redact_html"
    description: str = (
        "Redacts personally identifiable information (PII) from HTML content. "
        "Input should be an HTML string that may contain PII such as names, "
        "addresses, phone numbers, emails, or other sensitive data. "
        "Output is the HTML with PII entities redacted or replaced. "
        "For .html and .htm files, read the file contents and pass the HTML "
        "string to this tool. Do NOT use this tool for plain text, JSON, or "
        "binary files."
    )
    args_schema: type[BaseModel] = _RedactHtmlInput

    def _run(
        self,
        html_str: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Redact PII from the provided HTML content.

        Args:
            html_str: An HTML string to redact PII from.
            run_manager: The run manager for callbacks.

        Returns:
            The redacted HTML string with PII entities replaced.
        """
        if not html_str or not html_str.strip():
            return "Error: empty input. Provide an HTML string containing PII."
        try:
            json.loads(html_str)
            return (
                "Error: input looks like JSON, not HTML. "
                + _JSON_TOOL_REDIRECT
            )
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            response = self.client.redact_html(html_str, **self._build_kwargs())
            return response.redacted_text
        except Exception as e:
            return f"Error redacting HTML: {e}"


class TonicTextualRedactFile(_BaseTonicTextual):
    """Redact PII from files using Tonic Textual.

    Use this tool for binary and structured files: JPG, PNG, PDF, CSV, and TSV.
    The file is uploaded to Tonic Textual, redacted server-side, and the
    redacted file is written to ``output_path``.

    Do NOT use this tool for .txt, .json, .html, or .htm files. For those
    formats, read the file contents and pass them to ``TonicTextualRedactText``,
    ``TonicTextualRedactJson``, or ``TonicTextualRedactHtml`` respectively.

    Setup:
        Install ``langchain-textual`` and set environment variable
        ``TONIC_TEXTUAL_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-textual
            export TONIC_TEXTUAL_API_KEY="your-api-key"

    Instantiation:
        .. code-block:: python

            from langchain_textual import TonicTextualRedactFile

            tool = TonicTextualRedactFile()

    Invocation:
        .. code-block:: python

            tool.invoke({"file_path": "/path/to/scan.pdf"})
            tool.invoke({
                "file_path": "/path/to/photo.jpg",
                "output_path": "/path/to/photo_redacted.jpg",
            })
    """

    name: str = "tonic_textual_redact_file"
    description: str = (
        "Redacts personally identifiable information (PII) from files. "
        "Supported file types: JPG, PNG, PDF, CSV, TSV. "
        "Input is a file path to a file that may contain PII. "
        "The redacted file is written to output_path (defaults to "
        "<original_name>_redacted.<ext> in the same directory). "
        "Returns the path to the redacted file. "
        "Do NOT use this tool for .txt, .json, .html, or .htm files — "
        "for those, read the file and use the text, JSON, or HTML redaction "
        "tools instead."
    )
    args_schema: type[BaseModel] = _RedactFileInput

    def _run(
        self,
        file_path: str,
        output_path: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Redact PII from a file.

        Args:
            file_path: Path to the file to redact.
            output_path: Path to write the redacted file. Defaults to
                ``<original_name>_redacted.<ext>`` in the same directory.
            run_manager: The run manager for callbacks.

        Returns:
            The path to the redacted output file.
        """
        file_path = os.path.expanduser(file_path)
        _, ext = os.path.splitext(file_path)
        ext_lower = ext.lower()

        if ext_lower == ".txt":
            return (
                f"Error: {ext} files are not supported by this tool. "
                + _TEXT_TOOL_REDIRECT
            )
        if ext_lower == ".json":
            return (
                f"Error: {ext} files are not supported by this tool. "
                + _JSON_TOOL_REDIRECT
            )
        if ext_lower in {".html", ".htm"}:
            return (
                f"Error: {ext} files are not supported by this tool. "
                + _HTML_TOOL_REDIRECT
            )
        if ext_lower and ext_lower not in _SUPPORTED_FILE_EXTENSIONS:
            supported = ", ".join(sorted(_SUPPORTED_FILE_EXTENSIONS))
            return (
                f"Error: unsupported file type '{ext}'. "
                f"Supported extensions: {supported}."
            )

        if not os.path.exists(file_path):
            return f"Error: file not found: {file_path}"

        if output_path is None:
            base, _ = os.path.splitext(file_path)
            output_path = f"{base}_redacted{ext}"
        else:
            output_path = os.path.expanduser(output_path)

        file_name = os.path.basename(file_path)

        try:
            with open(file_path, "rb") as f:
                job_id = self.client.start_file_redaction(f, file_name)

            redacted_bytes = self.client.download_redacted_file(
                job_id, **self._build_kwargs()
            )

            with open(output_path, "wb") as f:
                f.write(redacted_bytes)

            return output_path
        except Exception as e:
            return f"Error redacting file: {e}"


class TonicTextualPiiTypes(BaseTool):
    """List all PII entity types supported by Tonic Textual.

    Use this tool to discover valid entity type names for use with
    ``generator_config``. No API key or network access is required.

    Instantiation:
        .. code-block:: python

            from langchain_textual import TonicTextualPiiTypes

            tool = TonicTextualPiiTypes()

    Invocation:
        .. code-block:: python

            tool.invoke("")
    """

    name: str = "tonic_textual_pii_types"
    description: str = (
        "Lists all PII entity types supported by Tonic Textual. "
        "Call this tool to discover valid entity type names (e.g. "
        "NAME_GIVEN, EMAIL_ADDRESS, CREDIT_CARD) that can be used in "
        "generator_config to control per-type redaction behavior. "
        "No input is required."
    )
    args_schema: type[BaseModel] = _PiiTypesInput

    def _run(
        self,
        query: str = "",
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Return all supported PII entity types.

        Args:
            query: Unused. Accepts any value for compatibility with tool calling.
            run_manager: The run manager for callbacks.

        Returns:
            A comma-separated list of supported PII entity type names.
        """
        return ", ".join(member.value for member in PiiType)
