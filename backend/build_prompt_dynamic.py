from pathlib import Path
from evidence_builder import build_evidence, build_context_text

def build_prompt(question):
    query_plan, topic_sources, all_sources = build_evidence(question)
    context = build_context_text(question, query_plan, topic_sources)

    prompt = f"""
You are a high-end Audio DSP and AI Research Paper Assistant.

You must answer the user question using ONLY the retrieved evidence below.

Rules:
1. Do not use previous/general knowledge as the source of truth.
2. Do not invent missing facts.
3. Use only the retrieved sources.
4. Cite claims using [SOURCE 1], [SOURCE 2], etc.
5. If evidence is missing or weak, clearly say: "The uploaded papers do not provide enough evidence for this part."
6. For algorithm questions, explain assumptions, equations/algorithm idea, strengths, weaknesses, complexity, real-time suitability, and use cases when evidence exists.
7. For comparison questions, include a table first.
8. For implementation questions, extract practical steps only from evidence.
9. Keep the answer technical, precise, and research-grade.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{context}
"""
    return prompt

if __name__ == "__main__":
    question = input("Ask your research question: ").strip()

    if not question:
        print("No question provided.")
        raise SystemExit

    prompt = build_prompt(question)

    output_path = Path("data/extracted/final_dynamic_prompt.txt")
    output_path.write_text(prompt, encoding="utf-8")

    print("\nPrompt created:")
    print(output_path)