# Example: LangChain Agent with PII Redaction

A self-contained example of a LangChain ReAct agent that uses Tonic Textual to redact PII from text and files. Supports OpenAI, Anthropic, and Google Gemini via [LiteLLM](https://docs.litellm.ai/).

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- A [Tonic Textual](https://textual.tonic.ai) API key
- An API key for at least one LLM provider

## Setup

```bash
cd examples/

export TONIC_TEXTUAL_API_KEY="your-textual-api-key"

# Set one or more of these depending on which model you want to use:
export OPENAI_API_KEY="your-openai-key"       # for gpt-4o-mini (default)
export ANTHROPIC_API_KEY="your-anthropic-key"  # for claude models
export GEMINI_API_KEY="your-gemini-key"        # for gemini models
```

## Run

Redact PII from text (uses gpt-4o-mini by default):

```bash
uv run agent.py "Redact this: My name is John Smith and my email is john@example.com"
```

Use a different model with `--model`:

```bash
uv run agent.py --model anthropic/claude-sonnet-4-20250514 "Redact this: Call me at 555-123-4567"
uv run agent.py --model gemini/gemini-2.0-flash "What PII types can you detect?"
```

Redact a PDF file (place a PDF in this directory first):

```bash
uv run agent.py "Redact the file sample.pdf"
```

The `--model` flag accepts any [LiteLLM model string](https://docs.litellm.ai/docs/providers).

## What's happening

The agent has access to three tools:

| Tool | Purpose |
|------|---------|
| `tonic_textual_redact_text` | Redact PII from plain text |
| `tonic_textual_redact_file` | Redact PII from files (PDF, JPG, PNG, CSV, TSV) |
| `tonic_textual_pii_types` | List all supported PII entity types |

The LLM decides which tool to call based on your prompt. You can watch the tool calls and results stream in real time.
