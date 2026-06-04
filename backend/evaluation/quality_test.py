import json
import time
from pathlib import Path

from backend.answering.answer_orchestrator import run_research_question
from backend.common.logger_config import suppress_output

QUESTIONS = [
    "Summarize the uploaded papers.",
    "Which methods are suitable for real-time speech enhancement?",
    "Compare neural and classical approaches in the uploaded papers.",
    "What are the main limitations mentioned in the papers?",
    "What metrics are used to evaluate speech enhancement?",
]

REPORT_PATH = Path("data/extracted/quality_test_report.json")


def warmup():
    print("Warming up retrieval engine...")
    with suppress_output():
        run_research_question("warmup query for speech enhancement")
    print("Warmup complete.")


def run_one(question: str):
    start = time.time()
    result = run_research_question(question)
    elapsed = time.time() - start

    return {
        "question": question,
        "mode": result.get("mode"),
        "sources": result.get("source_count"),
        "routes": result.get("route_count"),
        "time_seconds": round(elapsed, 2),
        "answer_path": result.get("answer_path"),
        "context_path": result.get("context_path"),
    }


def main():
    print("=" * 100)
    print("AUDIO RAG QUALITY TEST — WARM SPEED")
    print("=" * 100)

    warmup()

    results = []
    total_time = 0

    for i, question in enumerate(QUESTIONS, 1):
        print("\n" + "-" * 100)
        print(f"Question {i}: {question}")

        item = run_one(question)
        results.append(item)
        total_time += item["time_seconds"]

        print(f"Mode: {item['mode']}")
        print(f"Sources: {item['sources']}")
        print(f"Routes: {item['routes']}")
        print(f"Time: {item['time_seconds']}s")
        print(f"Output saved: {item['answer_path']}")

    avg_time = round(total_time / len(results), 2)

    report = {
        "average_warm_time_seconds": avg_time,
        "question_count": len(results),
        "results": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print(f"Average warm time: {avg_time}s")
    print(f"Report saved: {REPORT_PATH}")
    print("=" * 100)


if __name__ == "__main__":
    main()