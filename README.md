# langchain-textual

[![PyPI version](https://img.shields.io/pypi/v/langchain-textual)](https://pypi.org/project/langchain-textual/)
[![CI](https://github.com/tonicai/langchain-tonic-textual/actions/workflows/ci.yml/badge.svg)](https://github.com/tonicai/langchain-tonic-textual/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/tonicai/langchain-tonic-textual/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

PII redaction tools for LangChain, powered by [Tonic Textual](https://textual.tonic.ai).

Strip names, emails, addresses, and other sensitive data from text, JSON, HTML, and files before they hit your LLM — or on the way back out. Drop them into any LangChain chain or agent as standard tools.

## Installation

```bash
pip install langchain-textual
```

## Quick start

```bash
export TONIC_TEXTUAL_API_KEY="your-api-key"
```

```python
from langchain_textual import TonicTextualRedactText

tool = TonicTextualRedactText()
tool.invoke("My name is John Smith and my email is john@example.com.")
# "My name is [NAME_GIVEN_xxxx] [NAME_FAMILY_xxxx] and my email is [EMAIL_ADDRESS_xxxx]."
```

## Tools

| Tool | Input | Use for |
|------|-------|---------|
| `TonicTextualRedactText` | Plain text string | Raw text, `.txt` file contents |
| `TonicTextualRedactJson` | JSON string | Raw JSON, `.json` file contents |
| `TonicTextualRedactHtml` | HTML string | Raw HTML, `.html`/`.htm` file contents |
| `TonicTextualRedactFile` | File path | PDFs, images (JPG, PNG), CSVs, TSVs |
| `TonicTextualPiiTypes` | None | List all supported PII entity types |

### Text

```python
from langchain_textual import TonicTextualRedactText

tool = TonicTextualRedactText()
tool.invoke("My name is John Smith and my email is john@example.com.")
# "My name is [NAME_GIVEN_xxxx] [NAME_FAMILY_xxxx] and my email is [EMAIL_ADDRESS_xxxx]."
```

### JSON

```python
from langchain_textual import TonicTextualRedactJson

tool = TonicTextualRedactJson()
tool.invoke('{"name": "John Smith", "email": "john@example.com"}')
# '{"name": "[NAME_GIVEN_xxxx] [NAME_FAMILY_xxxx]", "email": "[EMAIL_ADDRESS_xxxx]"}'
```

### HTML

```python
from langchain_textual import TonicTextualRedactHtml

tool = TonicTextualRedactHtml()
tool.invoke("<p>Contact John Smith at john@example.com</p>")
# "<p>Contact [NAME_GIVEN_xxxx] [NAME_FAMILY_xxxx] at [EMAIL_ADDRESS_xxxx]</p>"
```

### Files

```python
from langchain_textual import TonicTextualRedactFile

tool = TonicTextualRedactFile()
tool.invoke({"file_path": "/path/to/scan.pdf"})
# "/path/to/scan_redacted.pdf"

tool.invoke({"file_path": "/path/to/photo.jpg", "output_path": "/tmp/redacted.jpg"})
# "/tmp/redacted.jpg"
```

For `.txt`, `.json`, and `.html`/`.htm` files, read the file contents and pass them to the corresponding text, JSON, or HTML tool instead.

## Configuration

All tools share the same configuration options.

**Synthesis mode** — replace PII with realistic fake data instead of placeholders:

```python
tool = TonicTextualRedactText(generator_default="Synthesis")
tool.invoke("Contact Jane Doe at jane.doe@example.com.")
# "Contact Maria Chen at maria.chen@gmail.com."
```

**Per-entity control** — set handling per PII type with `generator_config`:

```python
tool = TonicTextualRedactText(
    generator_default="Off",
    generator_config={
        "NAME_GIVEN": "Synthesis",
        "NAME_FAMILY": "Synthesis",
        "EMAIL_ADDRESS": "Redaction",
    },
)
tool.invoke("Contact Jane Doe at jane.doe@example.com.")
# "Contact Maria Chen at chen@[EMAIL_ADDRESS_xxxx]."
```

Use `TonicTextualPiiTypes` to list all supported entity type names:

```python
from langchain_textual import TonicTextualPiiTypes

TonicTextualPiiTypes().invoke("")
# "NUMERIC_VALUE, LANGUAGE, MONEY, ..., EMAIL_ADDRESS, NAME_GIVEN, NAME_FAMILY, ..."
```

**Self-hosted deployment:**

```python
tool = TonicTextualRedactText(tonic_textual_base_url="https://textual.your-company.com")
```

**Explicit API key** (instead of env var):

```python
tool = TonicTextualRedactText(tonic_textual_api_key="your-api-key")
```

## Using with a LangChain agent

Every tool in this package is a standard LangChain [tool](https://python.langchain.com/docs/concepts/tools/), so they work anywhere tools do. Give your agent whichever combination it needs:

```python
from langchain_textual import (
    TonicTextualRedactText,
    TonicTextualRedactJson,
    TonicTextualRedactFile,
)
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [TonicTextualRedactText(), TonicTextualRedactJson(), TonicTextualRedactFile()]
agent = create_react_agent(llm, tools)
```

## Development

```bash
# install dependencies
uv sync --group dev --group test --group lint --group typing

# install pre-commit hooks (auto-runs ruff on each commit)
uv tool install pre-commit
pre-commit install

# run unit tests
make test

# run integration tests (requires TONIC_TEXTUAL_API_KEY)
make integration_tests

# lint & format
make lint
make format
```

## License

MIT
