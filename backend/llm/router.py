from pathlib import Path
from backend.config import (
    ANSWER_PROVIDER,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    ANTHROPIC_DEEP_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_DEEP_MODEL,
)


SYSTEM_RULES = """
You are a high-end Audio DSP and AI Research Paper Assistant.

Hard rules:
1. Use ONLY the retrieved uploaded-paper evidence.
2. Do NOT use previous/general knowledge as the source of truth.
3. Do NOT invent unsupported details.
4. Cite technical claims using [SOURCE 1], [SOURCE 2], etc.
5. If evidence is missing, say: "The uploaded papers do not provide enough evidence for this part."
6. Give clear, technical, useful answers.
"""


def build_final_prompt(question: str, context: str) -> str:
    return f"""
{SYSTEM_RULES}

USER QUESTION:
{question}

RETRIEVED EVIDENCE FROM UPLOADED PAPERS:
{context}

Write the final answer.

Required format:
1. Direct answer
2. Evidence-backed explanation
3. Practical guidance / method selection / steps if applicable
4. Limitations or missing evidence
5. Final recommendation

Citation rules:
- Use citations like [SOURCE 1].
- Important technical claims must be cited.
- If evidence is not enough, say so clearly.
"""


def answer_manual(question: str, context: str) -> str:
    prompt = build_final_prompt(question, context)
    out = Path("data/extracted/latest_manual_prompt.txt")
    out.write_text(prompt, encoding="utf-8")
    return prompt


def answer_claude(question: str, context: str, deep=True) -> str:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is missing.")

    from anthropic import Anthropic

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    model = ANTHROPIC_DEEP_MODEL if deep else ANTHROPIC_MODEL

    response = client.messages.create(
        model=model,
        max_tokens=7000,
        temperature=0.1,
        system=SYSTEM_RULES,
        messages=[{"role": "user", "content": build_final_prompt(question, context)}],
    )

    return "\n".join(
        block.text for block in response.content
        if getattr(block, "type", None) == "text"
    ).strip()


def answer_openai(question: str, context: str, deep=True) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing.")

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    model = OPENAI_DEEP_MODEL if deep else OPENAI_MODEL

    response = client.responses.create(
        model=model,
        reasoning={"effort": "high" if deep else "medium"},
        input=[
            {"role": "system", "content": SYSTEM_RULES},
            {"role": "user", "content": build_final_prompt(question, context)},
        ],
    )

    return response.output_text.strip()


def answer(question: str, context: str, deep=True) -> dict:
    if ANSWER_PROVIDER == "manual":
        text = answer_manual(question, context)
        return {
            "mode": "manual",
            "text": text,
            "path": "data/extracted/latest_manual_prompt.txt",
        }

    if ANSWER_PROVIDER == "claude":
        text = answer_claude(question, context, deep=deep)
        return {
            "mode": "claude",
            "text": text,
            "path": None,
        }

    if ANSWER_PROVIDER == "openai":
        text = answer_openai(question, context, deep=deep)
        return {
            "mode": "openai",
            "text": text,
            "path": None,
        }

    raise RuntimeError(f"Unsupported ANSWER_PROVIDER={ANSWER_PROVIDER}")