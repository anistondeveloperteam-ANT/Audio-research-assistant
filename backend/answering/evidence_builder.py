import os
from collections import defaultdict

from backend.retrieval.query_planner import plan_queries
from backend.retrieval.hybrid_retrieve import hybrid_retrieve
from backend.common.logger_config import debug_print

PER_TOPIC_SOURCE_LIMIT = int(os.getenv("PER_TOPIC_SOURCE_LIMIT", "3"))
TOTAL_SOURCE_LIMIT = int(os.getenv("TOTAL_SOURCE_LIMIT", "16"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))

WEAK_SECTIONS = {
    "references",
    "acknowledgment",
    "acknowledgements",
    "appendix"
}


def contains_required_terms(text, required_terms):
    if not required_terms:
        return True

    text_lower = text.lower()
    return any(term.lower() in text_lower for term in required_terms)


def source_key(result):
    return (
        result.get("title"),
        result.get("section"),
        result.get("page_start"),
        result.get("page_end"),
        (result.get("text") or "")[:180],
    )


def is_weak_section(result):
    section = (result.get("section") or "").strip().lower()
    return section in WEAK_SECTIONS


def source_quality_bonus(result):
    section = (result.get("section") or "").lower()
    chunk_type = (result.get("chunk_type") or "").lower()

    bonus = 0.0

    if section in ["abstract", "introduction"]:
        bonus += 0.05

    if section in ["method", "methodology", "proposed method", "algorithm", "system model"]:
        bonus += 0.10

    if section in ["experimental setup", "experiments", "results", "evaluation", "discussion"]:
        bonus += 0.08

    if chunk_type in ["algorithm", "equation", "results_or_metrics"]:
        bonus += 0.07

    if is_weak_section(result):
        bonus -= 1.0

    return bonus


def sort_results_for_evidence(results):
    for r in results:
        base = float(r.get("rerank_score", 0.0))
        r["evidence_score"] = base + source_quality_bonus(r)

    return sorted(results, key=lambda x: x.get("evidence_score", 0.0), reverse=True)


def accept_result(result, required_terms, seen, paper_counts, max_per_paper):
    if is_weak_section(result):
        return False

    title = result.get("title") or ""
    text = result.get("text") or ""

    combined = " ".join([
        title,
        result.get("section") or "",
        result.get("concepts") or "",
        text
    ])

    if not contains_required_terms(combined, required_terms):
        return False

    key = source_key(result)
    if key in seen:
        return False

    if paper_counts.get(title, 0) >= max_per_paper:
        return False

    return True


def build_evidence(question, per_topic_k=None, total_limit=None):
    if per_topic_k is None:
        per_topic_k = PER_TOPIC_SOURCE_LIMIT

    if total_limit is None:
        total_limit = TOTAL_SOURCE_LIMIT

    query_plan = plan_queries(question)

    topic_sources = defaultdict(list)
    all_sources = []
    seen = set()

    debug_print("Query routes:", len(query_plan))

    for item in query_plan:
        topic = item["topic"]
        query = item["query"]
        required_terms = item.get("required_terms", [])

        results = hybrid_retrieve(query, top_k=RETRIEVAL_TOP_K)
        results = sort_results_for_evidence(results)

        paper_counts = {}
        accepted = []

        # Strict pass: diverse papers
        for result in results:
            if accept_result(result, required_terms, seen, paper_counts, max_per_paper=1):
                title = result.get("title") or ""
                key = source_key(result)

                paper_counts[title] = paper_counts.get(title, 0) + 1
                seen.add(key)
                accepted.append(result)
                all_sources.append(result)

                if len(accepted) >= per_topic_k:
                    break

        # Fallback pass: allow second source from same paper
        if len(accepted) < 2:
            for result in results:
                if accept_result(result, required_terms, seen, paper_counts, max_per_paper=2):
                    title = result.get("title") or ""
                    key = source_key(result)

                    paper_counts[title] = paper_counts.get(title, 0) + 1
                    seen.add(key)
                    accepted.append(result)
                    all_sources.append(result)

                    if len(accepted) >= per_topic_k:
                        break

        topic_sources[topic] = accepted

        if len(all_sources) >= total_limit:
            break

    return query_plan, topic_sources, all_sources[:total_limit]


def build_context_text(question, query_plan, topic_sources):
    lines = []

    lines.append("=" * 100)
    lines.append("USER QUESTION")
    lines.append("=" * 100)
    lines.append(question)
    lines.append("")

    lines.append("=" * 100)
    lines.append("RETRIEVAL STRATEGY")
    lines.append("=" * 100)
    lines.append("The system dynamically generated retrieval routes based on the user question intent.")
    lines.append("Evidence below was retrieved from uploaded papers only.")
    lines.append("")

    lines.append("=" * 100)
    lines.append("RETRIEVED EVIDENCE")
    lines.append("=" * 100)

    source_counter = 1

    for topic, sources in topic_sources.items():
        if not sources:
            continue

        lines.append("")
        lines.append("#" * 100)
        lines.append(f"TOPIC: {topic}")
        lines.append("#" * 100)

        for result in sources:
            label = f"SOURCE {source_counter}"
            result["source_label"] = label

            text = result.get("text") or ""

            lines.append("")
            lines.append(f"[{label}]")
            lines.append(f"Topic: {topic}")
            lines.append(f"Paper: {result.get('title')}")
            lines.append(f"Section: {result.get('section')}")
            lines.append(f"Pages: {result.get('page_start')}-{result.get('page_end')}")
            lines.append(f"Type: {result.get('chunk_type')}")
            lines.append(f"Concepts: {result.get('concepts')}")
            lines.append("")
            lines.append(text[:2400])
            lines.append("")

            source_counter += 1

    return "\n".join(lines)


if __name__ == "__main__":
    print("evidence_builder.py is a backend component.")
    print("Use the Streamlit app:")
    print("streamlit run frontend\\streamlit_app.py")