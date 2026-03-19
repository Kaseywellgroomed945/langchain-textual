# Announcing langchain-textual: PII Detection, Synthesis, and Entity Extraction for LangChain

Organizations building with AI are sitting on a paradox. The unstructured data that makes models useful — support tickets, clinical notes, chat logs, scanned documents — is the same data that's most heavily regulated. You can't fine-tune a model on patient records without addressing HIPAA. You can't build a RAG pipeline over customer communications without thinking about GDPR. You can't populate a test environment with production data without exposing PII to people who were never authorized to see it.

This isn't a problem limited to one workflow. It surfaces everywhere sensitive data meets AI:

- **Model training and fine-tuning**, where real-world data produces better models — but that data contains names, SSNs, and account numbers that can't appear in training sets.
- **RAG pipelines**, where documents are chunked, embedded, and stored in vector databases. PII persists at every layer: the source documents, the chunks, the embeddings, and the retrieval results that eventually reach the LLM.
- **LLM proxies**, where organizations need to scrub PII from prompts before they leave internal infrastructure and from completions before they reach end users.
- **Agent workflows**, where tools pull data from databases, APIs, and files — each step potentially surfacing and forwarding PII downstream.
- **Regulatory compliance** — HIPAA de-identification under both Safe Harbor and Expert Determination, GDPR right-to-erasure, PCI cardholder data requirements.
- **Lower environments**, where QA teams and developers need data that behaves like production without carrying the regulatory exposure.

The common thread across all of these is that regex and rule-based approaches aren't sufficient. A pattern like `\d{3}-\d{2}-\d{4}` will match anything that looks like a Social Security number — including dates, product codes, and case numbers. The question isn't whether a string matches a pattern. It's whether, in context, "her social is 123-45-6789" means something different from "case number 123-45-6789." That requires NER models that understand language, not string matching.

[Tonic Textual](https://tonic.ai/textual) provides exactly this — transformer-based PII detection and transformation across text, JSON, HTML, PDFs, images, and tabular data. With `langchain-textual`, those capabilities are now available as standard LangChain tools.

## What Tonic Textual brings to the table

It's worth understanding what Textual does before looking at the LangChain integration, because the integration inherits all of it.

Textual's NER model identifies 46+ entity types across 50+ languages. These aren't just the obvious ones like email addresses and phone numbers. The model detects names (given and family, separately), dates of birth, occupations, healthcare IDs, routing numbers, IP addresses, and more — with a confidence score for each detection.

What happens after detection is where Textual differentiates itself. There are three capabilities:

**Redaction** replaces detected PII with labeled placeholders:

```
Input:  "Contact John Smith at john@example.com"
Output: "Contact [NAME_GIVEN_a1b2] [NAME_FAMILY_c3d4] at [EMAIL_ADDRESS_e5f6]"
```

This is the safest option when the goal is to guarantee that no real PII appears in the output. The placeholders are tagged with their entity type and a consistent identifier, which means you can track which replacements correspond to the same original entity across a document.

**Synthesis** replaces PII with realistic fake data:

```
Input:  "Contact John Smith at john@example.com"
Output: "Contact Maria Chen at maria.chen@gmail.com"
```

This distinction matters more than it might seem at first glance. If you're training a language model on de-identified clinical notes, every `[NAME_GIVEN_xxxx]` placeholder is a token the model learns to predict — one that has no relationship to how real names appear in text. The model's understanding of sentence structure around names degrades. Synthesized data preserves the format and distribution of the original. A synthesized email is still a valid email. A synthesized date has the right format. Downstream models and analytics work correctly on the transformed data.

**Entity extraction** returns the raw detections without modifying the text — every PII entity found, with its type, value, location, and confidence score:

```
Input:  "Contact John Smith at john@example.com"
Output: [
  {"label": "NAME_GIVEN", "text": "John", "start": 8, "end": 12, "score": 0.9},
  {"label": "NAME_FAMILY", "text": "Smith", "start": 13, "end": 18, "score": 0.9},
  {"label": "EMAIL_ADDRESS", "text": "john@example.com", "start": 22, "end": 38, "score": 0.9}
]
```

This is useful when you need to know *what* PII is present without transforming it — for auditing, building custom downstream logic, populating metadata, or letting an agent reason about the data before deciding how to handle it.

For organizations navigating HIPAA, these capabilities are particularly relevant. Expert Determination requires a qualified statistical expert to certify that the risk of re-identification is very small. Synthesis that preserves statistical properties while eliminating re-identification risk is a powerful tool in that process — far more so than blanket redaction that destroys data utility.

Textual handles all of this across multiple formats. In JSON, it understands the difference between keys (typically not PII) and values (often PII). In HTML, it preserves markup structure while redacting text content. For PDFs and images, it uses OCR to detect text and then redacts in the rendered output. It runs in the cloud or self-hosted on your own infrastructure.

## The LangChain integration

`langchain-textual` wraps Textual's capabilities into six LangChain tools:

| Tool | Input | Use for |
|------|-------|---------|
| `TonicTextualRedactText` | Plain text string | Synthesize or tokenize PII in raw text, `.txt` file contents |
| `TonicTextualRedactJson` | JSON string | Synthesize or tokenize PII in raw JSON, `.json` file contents |
| `TonicTextualRedactHtml` | HTML string | Synthesize or tokenize PII in raw HTML, `.html`/`.htm` file contents |
| `TonicTextualRedactFile` | File path | Synthesize or tokenize PII in PDFs, images (JPG, PNG), CSVs, TSVs |
| `TonicTextualExtractEntities` | Plain text string | Extract detected PII entities with type, value, location, and confidence |
| `TonicTextualPiiTypes` | None | List all supported PII entity types |

These are standard LangChain `BaseTool` subclasses. They work with any agent, chain, or tool-calling model. Installation is a single line:

```bash
pip install langchain-textual
```

The separation into distinct tools is deliberate. When an LLM agent has access to multiple tools, it selects which one to call based on the tool's name, description, and input schema. A single tool that handles text, JSON, HTML, and files via a format parameter forces the model to make two decisions — "should I redact?" and "what format is this?" — when it should only need to make one. Separate tools with clear descriptions let the model match the user's intent to the right tool directly.

### How the tools are structured

All redaction tools share a common base class that handles client initialization and configuration:

```python
class _BaseTonicTextual(BaseTool):
    client: TextualNer = Field(default=None)
    tonic_textual_api_key: SecretStr = Field(default=SecretStr(""))
    tonic_textual_base_url: str | None = None
    generator_default: Literal["Off", "Redaction", "Synthesis"] | None = None
    generator_config: dict[str, Literal["Off", "Redaction", "Synthesis"]] = Field(
        default_factory=dict
    )
```

A Pydantic `model_validator` runs before instantiation, reading the API key from the constructor argument or the `TONIC_TEXTUAL_API_KEY` environment variable and initializing the Textual client. Each tool subclass then implements `_run` with a single call to the appropriate Textual API method — `client.redact()` for text, `client.redact_json()` for JSON, and so on.

Configuration options like `generator_default` and `generator_config` are defined on the base class, so every tool inherits them. A shared `_build_kwargs()` method assembles these into the keyword arguments that every Textual API method accepts. This keeps each tool's `_run` method focused on its format-specific logic.

### Teaching the agent which tool to use

When an LLM receives a tool definition, it sees three things: the tool's name, its description, and its input schema — including the `description` field on each parameter. These descriptions are instructions the model reads before deciding how to call the tool.

We define explicit Pydantic schemas for each tool's input:

```python
class _RedactTextInput(BaseModel):
    text: str = Field(
        description=(
            "Plain text that may contain PII. "
            "For .txt files, read the file first and pass the contents here."
        )
    )

class _RedactFileInput(BaseModel):
    file_path: str = Field(
        description=(
            "Absolute path to the file to redact. "
            "Supported types: JPG, PNG, PDF, CSV, TSV. "
            "Do NOT use for .txt, .json, .html, or .htm files — use the "
            "dedicated text, JSON, or HTML redaction tools instead."
        )
    )
```

By stating what each parameter expects and explicitly noting what *not* to use each tool for, we significantly reduce the chance the agent picks the wrong tool. The text tool says "Do NOT use this tool for JSON, HTML, or binary files." The file tool says "Do NOT use for .txt, .json, .html, or .htm files." These negative instructions are as important as the positive ones — models are good at following explicit boundaries.

### When the agent picks the wrong tool anyway

Even with good descriptions, agents sometimes make mistakes. The question is what happens next. If a tool returns a raw Python traceback, the agent has nothing to work with. If it returns an actionable error message, the agent can self-correct.

Each tool validates its input and redirects when appropriate:

```python
# In the text tool: if the input is valid JSON, redirect
try:
    json.loads(text)
    return (
        "Error: input looks like JSON, not plain text. "
        "Use tonic_textual_redact_json for .json files (read contents first)."
    )
except (json.JSONDecodeError, TypeError):
    pass
```

```python
# In the file tool: if the extension is .html, redirect
if ext_lower in {".html", ".htm"}:
    return (
        "Error: .htm files are not supported by this tool. "
        "Use tonic_textual_redact_html for .html/.htm files (read contents first)."
    )
```

This works because LLM agents treat tool outputs as observations in their reasoning loop. An error that names the correct tool is just another observation the agent can act on. In practice, the agent calls the right tool on the next step, and the user never sees the intermediate error.

## Putting it together: a working agent

Here's a complete agent with PII detection, transformation, and extraction capabilities:

```python
from langchain_textual import (
    TonicTextualRedactText,
    TonicTextualRedactFile,
    TonicTextualExtractEntities,
    TonicTextualPiiTypes,
)
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [
    TonicTextualRedactText(),
    TonicTextualRedactFile(),
    TonicTextualExtractEntities(),
    TonicTextualPiiTypes(),
]
agent = create_react_agent(llm, tools)
```

Two environment variables (`TONIC_TEXTUAL_API_KEY` and `OPENAI_API_KEY`), a few lines of code, and you have an agent that can detect, extract, and transform PII across text and files.

When a user asks "What PII is in this text: My name is John Smith and my email is john@example.com," the agent calls `tonic_textual_extract_entities`. It returns a JSON array of every detected entity with its type, value, position, and confidence score — giving the agent (or the user) full visibility into what was found.

When a user asks "Redact this: My name is John Smith and my email is john@example.com," the agent calls `tonic_textual_redact` with the text. Textual's NER model identifies the entities and the tool returns the transformed text.

When a user asks "Redact the file /tmp/medical_record.pdf," the agent calls `tonic_textual_redact_file`. Behind the scenes, the tool uploads the file to Textual, waits for the job to complete, downloads the result, and writes it to `/tmp/medical_record_redacted.pdf`. The agent reports back with the output path.

When a user asks "What PII types can you detect?" the agent calls `tonic_textual_pii_types`, which returns all 46 entity types from the SDK's `PiiType` enum — no API call needed, no latency.

## Controlling redaction at the entity level

The default behavior redacts everything Textual detects. But in practice, you often want different handling for different entity types. Names might be safe to synthesize for analytical purposes, while SSNs should always be hard-redacted. Credit card numbers get redacted; organization names might be left alone entirely.

`generator_config` provides this control:

```python
tool = TonicTextualRedactText(
    generator_default="Off",
    generator_config={
        "NAME_GIVEN": "Synthesis",
        "NAME_FAMILY": "Synthesis",
        "EMAIL_ADDRESS": "Redaction",
        "US_SSN": "Redaction",
    },
)
```

This configuration synthesizes names (so text reads naturally), hard-redacts emails and SSNs (for maximum safety), and leaves everything else untouched. The `generator_default` of `"Off"` means any entity type not listed in `generator_config` passes through unchanged.

This composability is essential for compliance workflows. HIPAA's Safe Harbor method specifies 18 identifier categories that must be removed or generalized. With `generator_config`, you can map each of those categories to the appropriate handling — redaction for direct identifiers, synthesis for quasi-identifiers where you need to preserve analytical utility, and `"Off"` for categories that don't apply to your data.

Not sure which entity types are available? That's what `TonicTextualPiiTypes` is for:

```python
from langchain_textual import TonicTextualPiiTypes
TonicTextualPiiTypes().invoke("")
# "NUMERIC_VALUE, LANGUAGE, MONEY, ..., US_SSN, CREDIT_CARD, EMAIL_ADDRESS, ..."
```

## What's next

This initial release covers PII transformation (synthesis and tokenization), entity extraction, and multi-format support (text, JSON, HTML, and binary files). We're continuing to expand format support and add capabilities.

The package is open source under the MIT license:

- **PyPI:** [langchain-textual](https://pypi.org/project/langchain-textual/)
- **GitHub:** [tonicai/langchain-tonic-textual](https://github.com/tonicai/langchain-tonic-textual)

A self-contained agent example lives in the `examples/` directory of the repo — clone it, set two environment variables, and run `uv run agent.py` to see it in action.

For the full Tonic Textual platform — including the web UI, dataset management, custom entity training, and enterprise deployment — visit [tonic.ai/textual](https://tonic.ai/textual).
