from pathlib import Path

from evidence_builder import build_evidence, build_context_text
from logger_config import debug_print


SYSTEM_RULES = """
You are a high-end Audio DSP and AI Research Paper Assistant.

Your answer must be:
- source-grounded
- technically accurate
- clear for a human user
- useful like a high-quality Claude/ChatGPT research answer

Hard rules:
1. Use ONLY the retrieved evidence.
2. Do NOT use previous/general knowledge as the source of truth.
3. Do NOT invent missing details.
4. Cite claims using [SOURCE 1], [SOURCE 2], etc.
5. If evidence is missing, say:
   "The uploaded papers do not provide enough evidence for this part."
6. Do not mention unretrieved papers or unsupported methods.
7. Do not expose backend retrieval details to the user.

Answer style:
1. Start with a direct answer.
2. Then give a structured explanation.
3. Use tables for comparisons or method selection.
4. For broad questions, give practical guidance grounded in evidence.
5. For algorithm questions, explain assumptions, method, strengths, weaknesses, metrics, and use cases if evidence supports it.
6. For implementation questions, provide steps only if the evidence supports them.
7. End with a final practical recommendation.
"""


def build_manual_prompt(question: str):
    debug_print("Building evidence from uploaded papers...")

    query_plan, topic_sources, all_sources = build_evidence(question)
    context = build_context_text(question, query_plan, topic_sources)

    prompt = f"""
{SYSTEM_RULES}

USER QUESTION:
{question}

RETRIEVED EVIDENCE FROM UPLOADED PAPERS:
{context}

Now write the final answer.

Required final answer format:
1. Direct answer
2. Evidence-backed explanation
3. Practical guidance / method selection / steps if applicable
4. Limitations or missing evidence
5. Final recommendation

Citation rules:
- Use citations like [SOURCE 1].
- Important technical claims must have citations.
- If the evidence is not enough, say so clearly.
"""

    context_path = Path("data/extracted/latest_context.txt")
    prompt_path = Path("data/extracted/latest_manual_prompt.txt")

    context_path.write_text(context, encoding="utf-8")
    prompt_path.write_text(prompt, encoding="utf-8")

    return {
        "context_path": str(context_path),
        "prompt_path": str(prompt_path),
        "source_count": len(all_sources),
    }


if __name__ == "__main__":
    print("manual_answer_engine.py is a backend component.")
    print("Use the Streamlit app:")
    print("streamlit run frontend\\streamlit_app.py")