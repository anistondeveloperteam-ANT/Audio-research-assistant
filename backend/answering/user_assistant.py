import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import time
from dotenv import load_dotenv

from backend.common.logger_config import suppress_output
from backend.answering.manual_answer_engine import build_manual_prompt

load_dotenv()

AUTO_INGEST = os.getenv("AUTO_INGEST", "true").lower() == "true"


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    print("=" * 80)
    print("HIGH-END AUDIO RESEARCH PAPER ASSISTANT")
    print("=" * 80)
    print("Ask anything about your uploaded papers.")
    print("Type 'exit' to quit.")
    print("")


def auto_update_index_quietly():
    """
    Silently checks new PDFs and embeds only missing chunks.
    """
    if not AUTO_INGEST:
        return

    try:
        with suppress_output():
            from auto_update_index import update_index
            update_index()
    except Exception:
        # Do not show backend errors to user mode.
        # Details are stored in data/logs/backend_silent.log
        pass


def warmup_retrieval_quietly():
    """
    Loads retrieval models silently once at startup.
    This reduces first-question delay and hides HF/model warnings.
    """
    try:
        with suppress_output():
            from backend.retrieval.hybrid_retrieve import hybrid_retrieve
            hybrid_retrieve("warmup query", top_k=1)
    except Exception:
        # Do not block app startup if warmup fails.
        pass


def create_prompt_quietly(question: str):
    with suppress_output():
        result = build_manual_prompt(question)
    return result


def open_prompt_file():
    try:
        os.system('notepad data\\extracted\\latest_manual_prompt.txt')
    except Exception:
        pass


def main():
    clear_screen()

    print("Starting assistant...")
    auto_update_index_quietly()
    warmup_retrieval_quietly()

    clear_screen()
    print_header()

    while True:
        question = input("Your question: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye.")
            break

        if not question:
            continue

        print("\nThinking...")

        start = time.time()

        try:
            result = create_prompt_quietly(question)
            elapsed = time.time() - start

            print("\nReady.")
            print(f"Evidence sources used: {result['source_count']}")
            print(f"Time taken: {elapsed:.2f} seconds")
            print("")
            print("Your evidence-grounded prompt is ready here:")
            print(result["prompt_path"])
            print("")
            print("Opening prompt now...")
            open_prompt_file()
            print("")
            print("Copy-paste it into Claude/ChatGPT web for now.")
            print("When paid API is added, this assistant will print the final answer directly.")
            print("")

        except Exception as e:
            print("\nSomething went wrong.")
            print("Error:", e)
            print("Check data/logs/backend_silent.log for backend details.")
            print("")


if __name__ == "__main__":
    main()