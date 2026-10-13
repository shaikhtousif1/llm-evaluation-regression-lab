"""Evaluate the same local RAG pipeline used by src/rag_app.py."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rag_app import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_PATH,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    TOP_K,
    chunk_text,
    embed_text,
    generate_answer,
    retrieve_chunks,
)

ROOT = Path(__file__).resolve().parents[1]
TEST_CASES = ROOT / "data" / "test_cases.json"


def contains_any(text: str, alternatives: list[str]) -> bool:
    lowered = text.casefold()
    return any(item.casefold() in lowered for item in alternatives)


def evaluate_case(case: dict, chunks: list[str], vectors: list[list[float]], top_k: int) -> dict:
    retrieved = retrieve_chunks(case["question"], chunks, vectors, top_k=top_k)
    expected_heading = case.get("expected_heading")
    retrieval_hit = (
        None
        if expected_heading is None
        else any(expected_heading.casefold() in item["text"].casefold() for item in retrieved)
    )
    context = "\n\n--- Retrieved chunk ---\n\n".join(
        f"[chunk={item['chunk_id']} similarity={item['similarity']:.3f}]\n{item['text']}"
        for item in retrieved
    )
    answer = generate_answer(case["question"], context)

    if case["answerable"]:
        groups = case["required_fact_groups"]
        groups_passed = [contains_any(answer, alternatives) for alternatives in groups]
        answer_fact_pass = all(groups_passed) if groups_passed else True
        abstention_pass = None
    else:
        terms = [
            "does not specify", "doesn't specify", "not specified",
            "does not mention", "not mentioned", "not covered",
            "cannot determine", "can't determine",
        ]
        groups_passed = None
        answer_fact_pass = None
        abstention_pass = contains_any(answer, terms)

    return {
        "id": case["id"],
        "question": case["question"],
        "expected_heading": expected_heading,
        "retrieved": retrieved,
        "retrieval_hit": retrieval_hit,
        "answer": answer,
        "required_fact_groups_passed": groups_passed,
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
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--output", default="reports/evaluation.json")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    document = DOCUMENT_PATH.read_text(encoding="utf-8")
    cases = json.loads(TEST_CASES.read_text(encoding="utf-8"))
    chunks = chunk_text(document)
    print(f"Source: {DOCUMENT_PATH.relative_to(ROOT)}")
    print(f"Chunks: {len(chunks)} | Tests: {len(cases)} | top_k: {args.top_k}")
    print(f"Chunk size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"Models: embeddings={EMBEDDING_MODEL}, generation={GENERATION_MODEL}")
    print("Embedding policy chunks using the shared RAG implementation...")
    vectors = [embed_text(chunk) for chunk in chunks]

    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']}: {case['question']}")
        result = evaluate_case(case, chunks, vectors, args.top_k)
        results.append(result)
        print(f"  retrieved: {[x['chunk_id'] for x in result['retrieved']]}")
        print(
            f"  retrieval_hit={result['retrieval_hit']} "
            f"answer_fact_pass={result['answer_fact_pass']} "
            f"abstention_pass={result['abstention_pass']}"
        )

    answerable = [r for r in results if r["answer_fact_pass"] is not None]
    unanswerable = [r for r in results if r["abstention_pass"] is not None]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_document": str(DOCUMENT_PATH.relative_to(ROOT)),
        "top_k": args.top_k,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "generation_model": GENERATION_MODEL,
        "metrics": {
            "retrieval": summarize([r["retrieval_hit"] for r in results]),
            "answer_fact": summarize([r["answer_fact_pass"] for r in answerable]),
            "abstention": summarize([r["abstention_pass"] for r in unanswerable]),
        },
        "cases": results,
        "metric_caveat": (
            "Literal string checks are imperfect proxies for semantic correctness. "
            "Inspect retrieved evidence and answers manually."
        ),
    }
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSummary")
    for name, metric in report["metrics"].items():
        rate = "n/a" if metric["rate"] is None else f"{metric['rate']:.0%}"
        print(f"  {name}: {metric['passed']}/{metric['total']} ({rate})")
    print(f"Report written to: {output}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, KeyError, FileNotFoundError) as exc:
        raise SystemExit(f"\nError: {exc}") from exc
