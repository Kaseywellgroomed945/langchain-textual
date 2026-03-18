"""LangChain agent with Tonic Textual PII redaction tools.

This example creates a ReAct agent that can:
- Redact PII from plain text
- Redact PII from PDF files
- List supported PII entity types

Supports OpenAI, Anthropic, and Google Gemini via LiteLLM.

Usage:
    uv run agent.py "Redact: John Smith, john@example.com"
    uv run agent.py --model anthropic/claude-sonnet-4-20250514 "Redact sample.pdf"
    uv run agent.py --model gemini/gemini-2.0-flash "What PII types?"
"""

from __future__ import annotations

import sys

from langchain_community.chat_models import ChatLiteLLM
from langgraph.prebuilt import create_react_agent

from langchain_textual import (
    TonicTextualPiiTypes,
    TonicTextualRedactFile,
    TonicTextualRedactText,
)

DEFAULT_MODEL = "gpt-4o-mini"


def main() -> None:
    args = sys.argv[1:]

    if not args or args == ["--help"]:
        print("Usage: uv run agent.py [--model MODEL] <prompt>")
        print()
        print("Models (any LiteLLM model string):")
        print("  gpt-4o-mini                       (default, requires OPENAI_API_KEY)")
        print("  anthropic/claude-sonnet-4-20250514 (requires ANTHROPIC_API_KEY)")
        print("  gemini/gemini-2.0-flash            (requires GEMINI_API_KEY)")
        print()
        print("Examples:")
        print(
            '  uv run agent.py "Redact: My name is John Smith, email john@example.com"'
        )
        print(
            "  uv run agent.py --model"
            ' anthropic/claude-sonnet-4-20250514 "Redact sample.pdf"'
        )
        print('  uv run agent.py --model gemini/gemini-2.0-flash "What PII types?"')
        sys.exit(1)

    model = DEFAULT_MODEL
    if args[0] == "--model":
        if len(args) < 3:
            print("Error: --model requires a model name and a prompt.")
            sys.exit(1)
        model = args[1]
        args = args[2:]

    prompt = " ".join(args)

    llm = ChatLiteLLM(model=model)
    tools = [
        TonicTextualRedactText(),
        TonicTextualRedactFile(),
        TonicTextualPiiTypes(),
    ]
    agent = create_react_agent(llm, tools)

    print(f"Model: {model}")
    print(f"Prompt: {prompt}\n")

    for chunk in agent.stream({"messages": [{"role": "user", "content": prompt}]}):
        if "agent" in chunk:
            for msg in chunk["agent"]["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"[tool call] {tc['name']}({tc['args']})")
                elif hasattr(msg, "content") and msg.content:
                    print(f"\nAgent: {msg.content}")
        if "tools" in chunk:
            for msg in chunk["tools"]["messages"]:
                content = msg.content if hasattr(msg, "content") else str(msg)
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f"[tool result] {preview}")


if __name__ == "__main__":
    main()
