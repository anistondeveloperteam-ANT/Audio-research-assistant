import os
import re
import sys
from typing import Dict, List

MAX_QUERY_ROUTES = int(os.getenv("MAX_QUERY_ROUTES", "5"))
DOMAIN_EXPANSIONS = {
    "noise": ["noise suppression", "denoising", "speech enhancement", "SNR", "background noise"],
    "voice": ["speech quality", "speech enhancement", "intelligibility", "naturalness", "noise reduction"],
    "reverb": ["reverberation", "dereverberation", "WPE", "late reverberation", "room acoustics"],
    "echo": ["acoustic echo cancellation", "AEC", "double-talk", "echo suppression"],
    "beamforming": ["MVDR", "LCMV", "GSC", "steering vector", "microphone array", "DOA"],
    "real time": ["latency", "low complexity", "real-time", "causal", "streaming"],
    "metrics": ["PESQ", "STOI", "SDR", "SNR", "WER", "MOS"],
    "deepfilter": ["DeepFilterNet", "deep filtering", "ERB", "DfNet", "speech enhancement"],
}
DOMAIN_TERMS = [
    # Beamforming / array processing
    "MVDR", "LCMV", "GSC", "DOA", "direction of arrival",
    "beamforming", "beamformer", "microphone array", "steering vector",
    "covariance matrix", "spatial covariance", "array processing",

    # Enhancement / noise / reverb
    "speech enhancement", "noise suppression", "noise reduction",
    "noise cancellation", "denoising", "audio denoising",
    "dereverberation", "reverberation", "late reverberation",
    "WPE", "weighted prediction error",

    # Neural methods
    "DNN", "CNN", "RNN", "Transformer", "RNNoise", "DeepFilterNet",
    "deep filtering", "mask", "mask-based", "ideal ratio mask",
    "complex mask", "encoder-decoder", "U-Net",

    # Echo / deployment
    "AEC", "acoustic echo cancellation", "echo cancellation",
    "real-time", "low latency", "edge device", "complexity",

    # Metrics
    "PESQ", "STOI", "SI-SDR", "SDR", "SNR", "WER", "MOS",
    "latency", "parameters", "MACs"
]

INTENT_PATTERNS = {
    "how_to": [
        "how", "how to", "steps", "procedure", "workflow", "pipeline",
        "implement", "implementation", "remove", "reduce", "improve",
        "build", "apply", "use"
    ],
    "comparison": [
        "compare", "comparison", "difference", "different", "vs", "versus",
        "better", "best", "tradeoff", "trade-off", "pros", "cons"
    ],
    "summary": [
        "summarize", "summary", "overview", "survey", "review",
        "main points", "key findings"
    ],
    "limitations": [
        "limitation", "limitations", "weakness", "drawback", "challenge",
        "problem", "failure", "issue", "risk"
    ],
    "algorithm": [
        "algorithm", "method", "architecture", "formula", "equation",
        "mathematical", "derive", "processing flow"
    ],
    "experiments": [
        "experiment", "experimental", "result", "results", "metric",
        "benchmark", "dataset", "performance", "evaluation"
    ],
    "recommendation": [
        "recommend", "which method", "which one", "suitable",
        "when to use", "choose", "selection"
    ],
}

STOPWORDS = {
    "what", "how", "why", "when", "where", "which", "who", "is", "are",
    "was", "were", "be", "been", "being", "the", "a", "an", "and", "or",
    "to", "for", "of", "in", "on", "with", "from", "by", "about", "using",
    "use", "can", "should", "does", "do", "did", "into", "explain",
    "compare", "difference", "between", "tell", "me", "give"
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def detect_intent(question: str) -> str:
    q = question.lower()

    for intent, patterns in INTENT_PATTERNS.items():
        if any(pattern in q for pattern in patterns):
            return intent

    return "general"


def extract_terms(question: str) -> List[str]:
    q = question.lower()
    found = []

    for term in DOMAIN_TERMS:
        if term.lower() in q:
            found.append(term)

    return list(dict.fromkeys(found))


def extract_keywords(question: str) -> List[str]:
    words = re.findall(r"[A-Za-z0-9\-]+", question)
    keywords = []

    for word in words:
        low = word.lower()
        if len(low) < 3:
            continue
        if low in STOPWORDS:
            continue
        keywords.append(word)

    return list(dict.fromkeys(keywords))[:10]


def infer_audio_expansion(question: str, keywords: List[str]) -> List[str]:
    """
    General audio-domain expansion.
    Not hardcoded to one question; expands common user language
    into technical retrieval language.
    """
    q = question.lower()
    expansions = []

    if any(x in q for x in ["noise", "noisy", "denoise", "remove noise", "background"]):
        expansions.extend([
            "speech enhancement",
            "noise suppression",
            "noise reduction",
            "denoising",
            "RNNoise",
            "DeepFilterNet",
            "DNN speech enhancement",
            "mask-based enhancement",
            "PESQ",
            "STOI",
            "real-time"
        ])

    if any(x in q for x in ["echo", "aec"]):
        expansions.extend([
            "acoustic echo cancellation",
            "AEC",
            "echo suppression",
            "multi-channel acoustic echo cancellation"
        ])

    if any(x in q for x in ["reverb", "reverberation", "dereverb", "room"]):
        expansions.extend([
            "dereverberation",
            "late reverberation",
            "WPE",
            "weighted prediction error",
            "deep filtering"
        ])

    if any(x in q for x in ["beam", "beamforming", "array", "microphone"]):
        expansions.extend([
            "MVDR",
            "LCMV",
            "GSC",
            "DOA",
            "steering vector",
            "covariance matrix",
            "microphone array"
        ])

    if any(x in q for x in ["real time", "realtime", "fast", "latency", "edge"]):
        expansions.extend([
            "real-time",
            "low latency",
            "complexity",
            "edge device",
            "efficient speech enhancement"
        ])

    expansions.extend(keywords)
    return list(dict.fromkeys(expansions))[:14]


def build_routes(question: str, intent: str, terms: List[str], keywords: List[str]) -> List[Dict]:
    expansion = infer_audio_expansion(question, keywords)
    expansion_text = " ".join(expansion)
    term_text = " ".join(terms)
    keyword_text = " ".join(keywords)

    routes = []

    # Route 1: direct evidence
    routes.append({
        "topic": "direct_evidence",
        "query": normalize(f"{question} {term_text} {keyword_text}"),
        "required_terms": [],
        "intent": intent,
        "priority": 1
    })

    # Route 2: method / algorithm route
    routes.append({
        "topic": "methods_algorithms",
        "query": normalize(f"{question} {expansion_text} method algorithm architecture processing pipeline assumptions"),
        "required_terms": [],
        "intent": intent,
        "priority": 2
    })

    # Route 3: evidence / experiments route
    routes.append({
        "topic": "experiments_results",
        "query": normalize(f"{question} {expansion_text} experiments results metrics PESQ STOI SDR SNR latency performance"),
        "required_terms": [],
        "intent": intent,
        "priority": 3
    })

    # Route 4: limitations / tradeoffs route
    routes.append({
        "topic": "limitations_tradeoffs",
        "query": normalize(f"{question} {expansion_text} limitations weaknesses tradeoffs robustness complexity real-time"),
        "required_terms": [],
        "intent": intent,
        "priority": 4
    })

    # Route 5: recommendation / practical route
    if intent in ["how_to", "recommendation", "comparison", "general"]:
        routes.append({
            "topic": "practical_recommendation",
            "query": normalize(f"{question} {expansion_text} practical recommendation use case selection implementation guidance"),
            "required_terms": [],
            "intent": intent,
            "priority": 5
        })

    # Term-specific route only when strong technical terms are explicitly present
    for term in terms[:3]:
        routes.append({
            "topic": f"term_{term}",
            "query": normalize(f"{term} {question} evidence method results limitations comparison"),
            "required_terms": [term],
            "intent": "term_focused",
            "priority": 6
        })

    return routes


def plan_queries(question: str, max_queries: int = None) -> List[Dict]:
    question = normalize(question)

    if not question:
        return []

    if max_queries is None:
        max_queries = MAX_QUERY_ROUTES

    intent = detect_intent(question)
    terms = extract_terms(question)
    keywords = extract_keywords(question)

    routes = build_routes(question, intent, terms, keywords)

    seen = set()
    unique = []

    for route in routes:
        query = route["query"].lower()
        if query in seen:
            continue
        seen.add(query)
        unique.append(route)

    return unique[:max_queries]


def explain_plan(question: str):
    """
    Developer-only diagnostic.
    Normal user/UI should never call this.
    """
    routes = plan_queries(question)

    print("\nDeveloper diagnostic: dynamic retrieval routes")
    print("This screen is not for end users.\n")

    for i, route in enumerate(routes, 1):
        print(f"{i}. {route['topic']} | intent={route['intent']}")
        print(f"   {route['query']}")


if __name__ == "__main__":
    if "--debug" not in sys.argv:
        print("query_planner.py is a backend component.")
        print("Run the app with:")
        print("streamlit run frontend\\streamlit_app.py")
        print("")
        print("For developer diagnostics:")
        print("python backend\\query_planner.py --debug")
        raise SystemExit

    while True:
        q = input("\nAsk developer test question: ").strip()
        if not q:
            break
        explain_plan(q)