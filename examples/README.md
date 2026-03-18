# Example: LangChain Agent with PII Redaction

A self-contained example of a LangChain ReAct agent that uses Tonic Textual to redact PII from text and files.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- A [Tonic Textual](https://textual.tonic.ai) API key
- An [OpenAI](https://platform.openai.com) API key

## Setup

```bash
cd examples/

export TONIC_TEXTUAL_API_KEY="your-textual-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

## Run

Redact PII from text:

```bash
uv run agent.py "Redact this: My name is John Smith and my email is john@example.com"
```

Redact a PDF file (place a PDF in this directory first):

```bash
uv run agent.py "Redact the file sample.pdf"
```

List supported PII entity types:

```bash
uv run agent.py "What PII types can you detect?"
```

## What's happening

The agent has access to three tools:

| Tool | Purpose |
|------|---------|
| `tonic_textual_redact_text` | Redact PII from plain text |
| `tonic_textual_redact_file` | Redact PII from files (PDF, JPG, PNG, CSV, TSV) |
| `tonic_textual_pii_types` | List all supported PII entity types |

The LLM decides which tool to call based on your prompt. You can watch the tool calls and results stream in real time.
