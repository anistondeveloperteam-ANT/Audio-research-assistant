from pathlib import Path
from backend.answering.evidence_builder import build_evidence, build_context_text
from backend.llm.router import answer


def run_research_question(question: str, deep: bool = True) -> dict:
    query_plan, topic_sources, all_sources = build_evidence(question)
    context = build_context_text(question, query_plan, topic_sources)

    context_path = Path("data/extracted/latest_context.txt")
    context_path.write_text(context, encoding="utf-8")

    llm_result = answer(question, context, deep=deep)

    answer_text = llm_result["text"]
    answer_path = Path("data/extracted/latest_answer.txt")
    answer_path.write_text(answer_text, encoding="utf-8")

    return {
        "question": question,
        "mode": llm_result["mode"],
        "answer_text": answer_text,
        "answer_path": str(answer_path),
        "context_path": str(context_path),
        "manual_prompt_path": llm_result.get("path"),
        "source_count": len(all_sources),
        "route_count": len(query_plan),
    }


if __name__ == "__main__":
    print("answer_orchestrator.py is a backend component.")
    print("Use:")
    print("streamlit run frontend\\streamlit_app.py")