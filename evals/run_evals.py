"""Run RAG cases and save inspectable results and Matplotlib diagnostics.

Run from the project root with ``python -m evals.run_evals``.
Expected answer facts are preserved for review; no accuracy score is inferred.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from evals.rag_workflow import RAGWorkflow


DEFAULT_DATASET = Path(__file__).with_name("golden_questions_v1.json")


def run_evaluation(
    workflow: RAGWorkflow, cases: Sequence[dict[str, Any]], *, limit: int = 10
) -> dict[str, Any]:
    """Run cases with an injected workflow and return JSON-compatible results.

    Service errors propagate so an incomplete run cannot look successful.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    if not cases:
        raise ValueError("At least one evaluation case is required")
    ids: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in ids:
            raise ValueError("Cases must have unique, nonempty string IDs")
        query = case.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Case {case_id} must have a nonempty query")
        ids.add(case_id)

    results = []
    for case in cases:
        started = perf_counter()
        result = workflow.answer(case["query"], limit=limit)
        results.append({
            "case": dict(case),
            "result": asdict(result),
            "latency_seconds": perf_counter() - started,
            "retrieved_document_count": len(result.retrieved_documents),
        })
    return {
        "configuration": {
            "search_mode": workflow.search_mode,
            "query_rewriting": workflow.query_rewriting,
            "limit": limit,
        },
        "cases": results,
    }


def save_report(report: dict[str, Any], output_dir: Path) -> None:
    """Save results.json and diagnostics.png without requiring a display."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rows = report["cases"]
    labels = [row["case"]["id"] for row in rows]
    positions = list(range(len(labels)))
    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(labels) * 0.7), 7))
    config = report["configuration"]
    fig.suptitle(
        f"RAG diagnostics: {config['search_mode']}, "
        f"query rewriting {'on' if config['query_rewriting'] else 'off'}"
    )
    axes[0].bar(positions, [row["latency_seconds"] for row in rows])
    axes[0].set_ylabel("End-to-end latency (seconds)")
    axes[1].bar(positions, [row["retrieved_document_count"] for row in rows])
    axes[1].set_ylabel("Retrieved passages")
    for ax in axes:
        ax.set_xticks(positions, labels, rotation=45, ha="right")
        ax.set_xlabel("Case")
    fig.tight_layout()
    try:
        fig.savefig(output_dir / "diagnostics.png", dpi=150)
    finally:
        plt.close(fig)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    source.add_argument("--query", help="Run one question instead of the dataset")
    parser.add_argument("--search-mode", choices=("hybrid", "sparse", "dense"), default="hybrid")
    parser.add_argument(
        "--query-rewriting", action=argparse.BooleanOptionalAction, default=True,
        help="Rewrite before retrieval; disable with --no-query-rewriting",
    )
    parser.add_argument("--limit", type=positive_int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("evals/results"))
    args = parser.parse_args(argv)
    if args.query is not None:
        cases = [{"id": "single-query", "query": args.query}]
    else:
        cases = json.loads(args.dataset.read_text(encoding="utf-8"))["cases"]
    workflow = RAGWorkflow(
        search_mode=args.search_mode, query_rewriting=args.query_rewriting
    )
    report = run_evaluation(workflow, cases, limit=args.limit)
    save_report(report, args.output_dir)
    if args.query is not None:
        print(report["cases"][0]["result"]["answer"])
    print(f"Saved {len(cases)} cases to {args.output_dir}")


if __name__ == "__main__":
    main()
