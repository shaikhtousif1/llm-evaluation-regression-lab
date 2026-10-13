"""Run a small, repeatable local RAG evaluation against Ollama."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "data" / "banking_policy.md"
TEST_CASES = ROOT / "data" / "test_cases.json"
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "all-minilm"
GENERATION_MODEL = "llama3.2:1b"


def post(endpoint: str, payload: dict) -> dict:
    request = Request(
        f"{OLLAMA_URL}{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            "Cannot reach Ollama at localhost:11434. Start Ollama and check its local API."
        ) from exc


def split_sections(document: str) -> list[dict[str, str]]:
    """Keep each Markdown H2 heading with its section text."""
    sections: list[dict[str, str]] = []
    heading = "Document Introduction"
    body: list[str] = []

    for line in document.splitlines():
        if line.startswith("## "):
            if body:
                sections.append({"heading": heading, "text": "\n".join(body).strip()})
            heading = line[3:].strip()
            body = [line]
        else:
            body.append(line)

    if body:
        sections.append({"heading": heading, "text": "\n".join(body).strip()})

    return [section for section in sections if section["text"]]


def embed(text: str) -> list[float]:
    result = post("/api/embeddings", {"model": EMBEDDING_MODEL, "prompt": text})
    return result["embedding"]


def cosine_scores(query_vector: list[float], vectors: list[list[float]]) -> list[float]:
    query = np.asarray(query_vector, dtype=np.float32)
    matrix = np.asarray(vectors, dtype=np.float32)
    denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    return np.divide(
        matrix @ query,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    ).tolist()


def generate_answer(question: str, context: str) -> str:
    result = post(
        "/api/chat",
        {
            "model": GENERATION_MODEL,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer the question using only the supplied fictional bank policy. "
                        "If the context does not specify the answer, say that it does not "
                        "specify it. Do not invent fees, thresholds, or guarantees."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Policy context:\n{context}\n\nQuestion: {question}",
                },
            ],
        },
    )
    return result.get("message", {}).get("content", "").strip()


def contains_any(text: str, alternatives: list[str]) -> bool:
    lowered = text.casefold()
    return any(item.casefold() in lowered for item in alternatives)


def evaluate_case(
    case: dict, sections: list[dict[str, str]], vectors: list[list[float]], top_k: int
) -> dict:
    query_vector = embed(case["question"])
    scores = cosine_scores(query_vector, vectors)
    ranked_indices = sorted(
        range(len(sections)), key=lambda index: scores[index], reverse=True
    )[:top_k]
    retrieved = [sections[index] for index in ranked_indices]

    expected_heading = case.get("expected_heading")
    retrieval_hit = (
        None
        if expected_heading is None
        else any(section["heading"] == expected_heading for section in retrieved)
    )
    context = "\n\n--- Retrieved section ---\n\n".join(
        section["text"] for section in retrieved
    )
    generated = generate_answer(case["question"], context)

    if case["answerable"]:
        groups = case["required_fact_groups"]
        passed_groups = [contains_any(generated, alternatives) for alternatives in groups]
        answer_fact_pass = all(passed_groups) if passed_groups else True
        abstention_pass = None
    else:
        abstention_terms = [
            "does not specify", "doesn't specify", "not specified",
            "does not mention", "not mentioned", "not covered",
            "cannot determine", "can't determine",
        ]
        abstention_pass = contains_any(generated, abstention_terms)
        answer_fact_pass = None
        passed_groups = None

    return {
        "id": case["id"],
        "question": case["question"],
        "expected_heading": expected_heading,
        "retrieved": [
            {"heading": sections[index]["heading"], "similarity": round(scores[index], 4)}
            for index in ranked_indices
        ],
        "retrieval_hit": retrieval_hit,
        "answer": generated,
        "required_fact_groups_passed": passed_groups,
        "answer_fact_pass": answer_fact_pass,
        "abstention_pass": abstention_pass,
    }


def summarize(values: list[bool | None]) -> dict:
    observed = [value for value in values if value is not None]
    if not observed:
        return {"passed": 0, "total": 0, "rate": None}
    passed = sum(observed)
    return {"passed": passed, "total": len(observed), "rate": round(passed / len(observed), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default="reports/evaluation.json")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    document = DOCUMENT.read_text(encoding="utf-8")
    cases = json.loads(TEST_CASES.read_text(encoding="utf-8"))
    sections = split_sections(document)

    print(f"Sections: {len(sections)} | Test cases: {len(cases)} | top_k: {args.top_k}")
    print("Embedding policy sections...")
    vectors = [embed(section["text"]) for section in sections]

    results = []
    for number, case in enumerate(cases, start=1):
        print(f"[{number}/{len(cases)}] {case['id']}: {case['question']}")
        result = evaluate_case(case, sections, vectors, args.top_k)
        results.append(result)
        print(f"  retrieved: {[item['heading'] for item in result['retrieved']]}")
        print(
            f"  retrieval_hit={result['retrieval_hit']} "
            f"answer_fact_pass={result['answer_fact_pass']} "
            f"abstention_pass={result['abstention_pass']}"
        )

    answerable = [result for result in results if result["answer_fact_pass"] is not None]
    unanswerable = [result for result in results if result["abstention_pass"] is not None]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "top_k": args.top_k,
        "embedding_model": EMBEDDING_MODEL,
        "generation_model": GENERATION_MODEL,
        "metrics": {
            "retrieval": summarize([result["retrieval_hit"] for result in results]),
            "answer_fact": summarize([result["answer_fact_pass"] for result in answerable]),
            "abstention": summarize([result["abstention_pass"] for result in unanswerable]),
        },
        "cases": results,
        "metric_caveat": (
            "Simple deterministic string checks are imperfect proxies for semantic correctness. "
            "Review failed cases and generated answers manually."
        ),
    }

    output_path = (ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSummary")
    for name, metric in report["metrics"].items():
        rate_text = "n/a" if metric["rate"] is None else f"{metric['rate']:.0%}"
        print(f"  {name}: {metric['passed']}/{metric['total']} ({rate_text})")
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, KeyError, FileNotFoundError) as exc:
        raise SystemExit(f"\nError: {exc}") from exc
