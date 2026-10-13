"""Compare two evaluation JSON reports and show regressions."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_report(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric_rate(report: dict, name: str) -> float | None:
    return report["metrics"][name]["rate"]


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python src/compare.py <baseline-report.json> <candidate-report.json>"
        )

    baseline = load_report(sys.argv[1])
    candidate = load_report(sys.argv[2])

    print(
        f"Baseline top_k={baseline.get('top_k')} -> "
        f"Candidate top_k={candidate.get('top_k')}"
    )
    print("\nMetric comparison:")
    for name in ("retrieval", "answer_fact", "abstention"):
        before = metric_rate(baseline, name)
        after = metric_rate(candidate, name)
        if before is None or after is None:
            print(f"  {name}: not comparable")
        else:
            print(f"  {name}: {before:.0%} -> {after:.0%} ({after - before:+.0%})")

    old_cases = {case["id"]: case for case in baseline["cases"]}
    new_cases = {case["id"]: case for case in candidate["cases"]}
    print("\nPer-case regressions (pass -> fail):")
    found = False
    for case_id in sorted(old_cases.keys() & new_cases.keys()):
        for field in ("retrieval_hit", "answer_fact_pass", "abstention_pass"):
            if old_cases[case_id].get(field) is True and new_cases[case_id].get(field) is False:
                print(f"  {case_id}: {field}")
                found = True
    if not found:
        print("  None detected by these checks.")

    print("\nThese are simple regression signals, not a complete semantic quality judgment.")
    print("Inspect answers and retrieved sections for failures and surprising passes.")


if __name__ == "__main__":
    main()
