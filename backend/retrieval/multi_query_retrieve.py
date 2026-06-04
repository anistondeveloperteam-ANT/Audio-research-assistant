from collections import defaultdict
from backend.retrieval.hybrid_retrieve import hybrid_retrieve

SUBQUERIES = [
    {
        "topic": "MVDR",
        "query": "MVDR minimum variance distortionless response beamforming covariance matrix steering vector assumptions strengths weaknesses",
        "required_terms": ["MVDR", "beamforming", "covariance"],
    },
    {
        "topic": "LCMV",
        "query": "LCMV linearly constrained minimum variance beamforming multiple constraints distortionless response comparison with MVDR",
        "required_terms": ["LCMV", "linearly constrained", "constraint"],
    },
    {
        "topic": "GSC",
        "query": "GSC generalized sidelobe canceller beamforming blocking matrix adaptive noise canceller comparison with MVDR LCMV",
        "required_terms": ["GSC", "generalized sidelobe", "beamforming"],
    },
    {
        "topic": "DNN Speech Enhancement",
        "query": "DNN speech enhancement RNNoise DeepFilterNet noise suppression PESQ STOI real time low latency strengths weaknesses",
        "required_terms": ["DNN", "RNNoise", "DeepFilterNet", "speech enhancement"],
    },
    {
        "topic": "Dereverberation",
        "query": "speech dereverberation WPE weighted prediction error deep filtering late reverberation real time strengths weaknesses",
        "required_terms": ["dereverberation", "WPE", "late reverberation", "deep filtering"],
    },
]

def text_contains_any(text, terms):
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in terms)

def source_key(result):
    return (
        result["title"],
        result["page_start"],
        result["page_end"],
        result["section"],
        result["text"][:100],
    )

def run_planned_retrieval(per_topic_k=5):
    all_sources = []
    topic_sources = defaultdict(list)
    seen = set()

    for item in SUBQUERIES:
        topic = item["topic"]
        query = item["query"]
        required_terms = item["required_terms"]

        print("=" * 100)
        print("TOPIC:", topic)
        print("QUERY:", query)
        print("=" * 100)

        results = hybrid_retrieve(query, top_k=8)

        accepted = []

        for result in results:
            combined_text = " ".join([
                result.get("title") or "",
                result.get("section") or "",
                result.get("concepts") or "",
                result.get("text") or "",
            ])

            # Prefer real method chunks. Do not allow References to dominate.
            section = (result.get("section") or "").lower()
            if section == "references":
                continue

            # Accept if topic terms are actually present.
            if not text_contains_any(combined_text, required_terms):
                continue

            key = source_key(result)
            if key in seen:
                continue

            seen.add(key)
            accepted.append(result)
            all_sources.append(result)

            print(f"Accepted: {result['title']} | {result['section']} | pages {result['page_start']}-{result['page_end']}")

            if len(accepted) >= per_topic_k:
                break

        topic_sources[topic] = accepted

        if not accepted:
            print(f"WARNING: No strong source found for {topic}. You may need more papers for this topic.")

    return topic_sources, all_sources

def build_report(topic_sources):
    lines = []
    lines.append("=" * 100)
    lines.append("MULTI-QUERY RETRIEVAL REPORT")
    lines.append("=" * 100)
    lines.append("")

    for topic, sources in topic_sources.items():
        lines.append("")
        lines.append("#" * 100)
        lines.append(f"TOPIC: {topic}")
        lines.append("#" * 100)

        if not sources:
            lines.append("NO STRONG SOURCE FOUND FOR THIS TOPIC.")
            lines.append("Action: Add more papers specifically covering this method.")
            continue

        for i, r in enumerate(sources, 1):
            lines.append("")
            lines.append(f"[{topic} SOURCE {i}]")
            lines.append(f"Paper: {r['title']}")
            lines.append(f"Section: {r['section']}")
            lines.append(f"Pages: {r['page_start']}-{r['page_end']}")
            lines.append(f"Type: {r['chunk_type']}")
            lines.append(f"Concepts: {r['concepts']}")
            lines.append(f"Rerank score: {r.get('rerank_score', 0):.4f}")
            lines.append("")
            lines.append(r["text"][:2200])
            lines.append("")

    return "\n".join(lines)

if __name__ == "__main__":
    topic_sources, all_sources = run_planned_retrieval(per_topic_k=4)

    report = build_report(topic_sources)

    out_path = "data/extracted/multi_query_context.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\nSaved multi-query context to:")
    print(out_path)

    print("\nSummary:")
    for topic, sources in topic_sources.items():
        print(f"{topic}: {len(sources)} sources")