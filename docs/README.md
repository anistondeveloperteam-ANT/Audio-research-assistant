# 📚 Documentation index

Start with the root [README](../README.md) for the overview and quick start. These docs go deeper —
each has one clear purpose (no overlapping rewrites).

## How it works
- **[PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)** — the complete, diagram-first walkthrough of how a question
  becomes a verified, cited answer (and how a coding task becomes tested code). PDF-ready. **Start here.**
- [ARCHITECTURE.md](ARCHITECTURE.md) — system architecture, components, and the code map.
- [TECH_STACK.md](TECH_STACK.md) — the technology behind each piece and why it was chosen.

## Measured results
- [HOW_IT_WORKS.md](HOW_IT_WORKS.md) — measured accuracy + latency, stage by stage.
- [MEASUREMENT.md](MEASUREMENT.md) — the routing/guard classifier metrics + confusion matrices.
- [RAG_BASELINE.md](RAG_BASELINE.md) — the retrieval-quality benchmark.
- [CRAG_GRADING.md](CRAG_GRADING.md) — how the Corrective-RAG evidence grader is scored.

## Features & setup
- [LOGIN_SCREEN.md](LOGIN_SCREEN.md) — the accounts / login screen.
- [GOOGLE_SIGNIN.md](GOOGLE_SIGNIN.md) — optional "Continue with Google" (OAuth).
- [INGESTION_CHECKLIST.md](INGESTION_CHECKLIST.md) — adding your own PDFs to the library.
- [OBSERVABILITY.md](OBSERVABILITY.md) — optional Langfuse tracing / quality gates.
- [TURBOVEC_ACCELERATOR.md](TURBOVEC_ACCELERATOR.md) — the optional compressed local vector backend.

## Design specs (internal)
- [superpowers/](superpowers/) — design specs and plans for individual features.
