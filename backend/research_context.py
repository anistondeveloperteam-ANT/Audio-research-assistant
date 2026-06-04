from hybrid_retrieve import hybrid_retrieve

def build_context(question, results):
    lines = []

    lines.append("=" * 100)
    lines.append("QUESTION")
    lines.append("=" * 100)
    lines.append(question)
    lines.append("")

    lines.append("=" * 100)
    lines.append("RETRIEVED SOURCES")
    lines.append("=" * 100)

    for i, r in enumerate(results, 1):
        lines.append("")
        lines.append(f"[SOURCE {i}]")
        lines.append(f"Paper: {r['title']}")
        lines.append(f"Section: {r['section']}")
        lines.append(f"Pages: {r['page_start']}-{r['page_end']}")
        lines.append(f"Chunk type: {r['chunk_type']}")
        lines.append(f"Concepts: {r['concepts']}")
        lines.append(f"Rerank score: {r.get('rerank_score', 0):.4f}")
        lines.append("")
        lines.append(r["text"][:2500])
        lines.append("")

    return "\n".join(lines)

if __name__ == "__main__":
    question = "Compare MVDR, LCMV, GSC, DNN speech enhancement, and dereverberation methods. Explain assumptions, strengths, weaknesses, and when to use each."

    results = hybrid_retrieve(question, top_k=10)
    context = build_context(question, results)

    output_path = "data/extracted/first_research_context.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(context)

    print("\nResearch context saved to:")
    print(output_path)

    print("\nPreview:\n")
    print(context[:4000])