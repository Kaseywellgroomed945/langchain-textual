"""LangChain agent with Tonic Textual PII redaction tools.

This example creates a ReAct agent that can:
- Redact PII from plain text
- Redact PII from PDF files
- List supported PII entity types

Usage:
    uv run agent.py "Redact this: My name is John Smith and my email is john@example.com"
    uv run agent.py "Redact the file sample.pdf"
    uv run agent.py "What PII types can you detect?"
"""

from __future__ import annotations

import sys

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from langchain_textual import (
    TonicTextualPiiTypes,
    TonicTextualRedactFile,
    TonicTextualRedactText,
)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run agent.py <prompt>")
        print()
        print("Examples:")
        print('  uv run agent.py "Redact: My name is John Smith, email john@example.com"')
        print('  uv run agent.py "Redact the file sample.pdf"')
        print('  uv run agent.py "What PII types are supported?"')
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])

    llm = ChatOpenAI(model="gpt-4o-mini")
    tools = [
        TonicTextualRedactText(),
        TonicTextualRedactFile(),
        TonicTextualPiiTypes(),
    ]
    agent = create_react_agent(llm, tools)

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
