"""
DeepEval LLM-quality gates: faithfulness, answer relevancy, and contextual relevancy,
evaluated over the fixtures in tests/fixtures/llm_quality.json. Each metric must clear 0.7.

OPT-IN: these run only when DEEPEVAL_ENABLED=true AND a judge-model key is configured.
Otherwise they skip cleanly, so the normal offline suite is unaffected.

Judge model (an LLM grades the answers) — configure ONE, reusing the app's OpenAI-compatible
provider style:
  * DEEPEVAL_JUDGE_API_KEY  (+ optional DEEPEVAL_JUDGE_BASE_URL, DEEPEVAL_JUDGE_MODEL), or
  * a fallback to OPENAI_CLOUD_KEY / OPENAI_API_KEY (+ optional OPENAI_BASE_URL / OPENAI_MODEL).
DeepEval's GPTModel accepts base_url, so any OpenAI-compatible endpoint (incl. a local
Ollama at http://localhost:11434/v1) works as the judge.

    DEEPEVAL_ENABLED=true DEEPEVAL_JUDGE_API_KEY=sk-... \
        .venv/Scripts/python.exe -m pytest tests/test_llm_quality.py -v
"""
import json
import os
from pathlib import Path

import pytest

_ENABLED = os.getenv("DEEPEVAL_ENABLED", "").strip().lower() == "true"
_FIXTURES = Path(__file__).parent / "fixtures" / "llm_quality.json"
THRESHOLD = 0.7


def _judge_config():
    """(model, api_key, base_url) resolved from env; api_key is None when unset."""
    api_key = (os.getenv("DEEPEVAL_JUDGE_API_KEY") or os.getenv("OPENAI_CLOUD_KEY")
               or os.getenv("OPENAI_API_KEY"))
    base_url = os.getenv("DEEPEVAL_JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    model = os.getenv("DEEPEVAL_JUDGE_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    return model, api_key, base_url


_MODEL, _KEY, _BASE = _judge_config()

pytestmark = pytest.mark.skipif(
    not _ENABLED or not _KEY,
    reason="LLM-quality gates are opt-in: set DEEPEVAL_ENABLED=true and a judge-model "
           "key (see this file's docstring).")


def _judge():
    from deepeval.models import GPTModel
    kwargs = {"model": _MODEL, "api_key": _KEY}
    if _BASE:
        kwargs["base_url"] = _BASE
    return GPTModel(**kwargs)


def _cases():
    from deepeval.test_case import LLMTestCase
    data = json.loads(_FIXTURES.read_text(encoding="utf-8"))
    return [
        LLMTestCase(input=d["question"], actual_output=d["answer"],
                    retrieval_context=list(d["context"]))
        for d in data
    ]


def _assert_metric(metric_cls, needs_context=True):
    pytest.importorskip("deepeval")
    metric = metric_cls(threshold=THRESHOLD, model=_judge(), async_mode=False)
    for tc in _cases():
        metric.measure(tc)
        assert metric.score is not None and metric.score >= THRESHOLD, (
            f"{metric_cls.__name__} scored {metric.score} (< {THRESHOLD}) "
            f"for {tc.input!r}: {getattr(metric, 'reason', '')}")


def test_faithfulness():
    from deepeval.metrics import FaithfulnessMetric
    _assert_metric(FaithfulnessMetric)


def test_answer_relevancy():
    from deepeval.metrics import AnswerRelevancyMetric
    _assert_metric(AnswerRelevancyMetric)


def test_contextual_relevancy():
    from deepeval.metrics import ContextualRelevancyMetric
    _assert_metric(ContextualRelevancyMetric)
