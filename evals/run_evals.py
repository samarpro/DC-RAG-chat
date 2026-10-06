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
HIT_KS = (3, 5)


def file_name(source: str) -> str:
    """Compare exact basenames, allowing Unix and Windows source paths."""
    return source.strip().replace("\\", "/").rsplit("/", 1)[-1]


def retrieval_metrics(
    reference_files: Sequence[str], ranked_sources: Sequence[str | None]
) -> dict[str, Any]:
    """Score passage ranks against reference filenames without deduplicating ranks."""
    metrics = {"status": "no_references",
               "hit_at_k": {str(k): {"any": None, "all": None} for k in HIT_KS},
               "first_relevant_rank": None,
               "reciprocal_rank_at_limit": None}
    if not reference_files:
        return metrics
    if any(source is None or not source.strip() for source in ranked_sources):
        metrics["status"] = "missing_source"
        return metrics
    references = {file_name(source) for source in reference_files}
    sources = [file_name(source) for source in ranked_sources]
    first_rank = next((rank for rank, source in enumerate(sources, 1)
                       if source in references), None)
    return {
        "status": "scored",
        "hit_at_k": {
            str(k): {"any": int(bool(references & set(sources[:k]))),
                     "all": int(references <= set(sources[:k]))}
            for k in HIT_KS
        },
        "first_relevant_rank": first_rank,
        "reciprocal_rank_at_limit": 1 / first_rank if first_rank else 0.0,
    }


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
        references = case.get("reference_files", [])
        if not isinstance(references, list) or any(
            not isinstance(source, str) or not file_name(source) for source in references
        ):
            raise ValueError(f"Case {case_id} reference_files must be a list of nonempty filenames")
        if references and limit < max(HIT_KS):
            raise ValueError("limit must be at least 5 when scoring reference files")
        ids.add(case_id)

    results = []
    for case in cases:
        started = perf_counter()
        result = workflow.answer(case["query"], limit=limit)
        latency = perf_counter() - started
        results.append({
            "case": dict(case),
            "result": asdict(result),
            "latency_seconds": latency,
            "retrieval_metrics": retrieval_metrics(
                case.get("reference_files", []), result.ranked_sources
            ),
        })
    return {
        "configuration": {
            "search_mode": workflow.search_mode,
            "query_rewriting": workflow.query_rewriting,
            "limit": limit,
            "hit_ks": list(HIT_KS),
        },
        "cases": results,
        "retrieval_summary": summarize_retrieval(results),
    }


def summarize_retrieval(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scored = [row["retrieval_metrics"] for row in results
              if row["retrieval_metrics"]["status"] == "scored"]
    summary = {"scored_cases": len(scored), "excluded_cases": len(results) - len(scored)}
    summary["hit_at_k"] = {
        str(k): {kind: sum(row["hit_at_k"][str(k)][kind] for row in scored) / len(scored)
                 if scored else None for kind in ("any", "all")}
        for k in HIT_KS
    }
    summary["mrr_at_limit"] = (
        sum(row["reciprocal_rank_at_limit"] for row in scored) / len(scored) if scored else None
    )
    return summary


def save_report(report: dict[str, Any], output_dir: Path) -> None:
    """Save JSON, paired hit@3/5 panels, latency, and dataset MRR plots."""
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
    config = report["configuration"]
    caption = (
        f"{config['search_mode']}, "
        f"query rewriting {'on' if config['query_rewriting'] else 'off'}"
    )
    width = max(8, len(labels) * 0.7)
    scored = [(position, row["retrieval_metrics"]) for position, row in enumerate(rows)
              if row["retrieval_metrics"]["status"] == "scored"]
    summary = report["retrieval_summary"]

    def label_cases(ax):
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_xticks(positions, labels, rotation=45, ha="right")
        ax.set_xlabel("Case")

    def save_figure(fig, name):
        try:
            fig.tight_layout()
            fig.savefig(output_dir / name, dpi=150)
        finally:
            plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(width, 8))
    fig.suptitle(f"Reference file hits: {caption}")
    for ax, k in zip(axes, HIT_KS):
        if scored:
            for offset, kind, label in ((-0.2, "any", "Any reference"), (0.2, "all", "All references")):
                ax.bar([position + offset for position, _ in scored],
                       [metrics["hit_at_k"][str(k)][kind] for _, metrics in scored],
                       width=0.4, label=label)
            means = summary["hit_at_k"][str(k)]
            ax.set_title(f"Mean hit@{k}: any {means['any']:.3f}, all {means['all']:.3f}")
            ax.legend()
        else:
            ax.set_title(f"Hit@{k}: no scored cases")
        ax.set_ylabel(f"Hit@{k}")
        ax.set_ylim(0, 1.15)
        for position, row in enumerate(rows):
            if row["retrieval_metrics"]["status"] != "scored":
                ax.text(position, 0.05, "N/A", ha="center", fontsize=8)
        label_cases(ax)
    save_figure(fig, "hit_at_3_and_5.png")

    fig, ax = plt.subplots(figsize=(width, 4))
    ax.bar(positions, [row["latency_seconds"] for row in rows])
    ax.set_title(f"End-to-end latency: {caption}")
    ax.set_ylabel("Latency (seconds)")
    label_cases(ax)
    save_figure(fig, "latency.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    mrr = summary["mrr_at_limit"]
    if mrr is not None:
        ax.bar([0], [mrr], width=0.5)
        ax.text(0, mrr + 0.03, f"{mrr:.3f}", ha="center")
    else:
        ax.text(0, 0.5, "N/A: no scored cases", ha="center")
    ax.set_xticks([0], [f"{summary['scored_cases']} scored cases"])
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(f"MRR@{config['limit']}")
    ax.set_title(f"Dataset MRR@{config['limit']}: {caption}", fontsize=10)
    save_figure(fig, "mrr.png")


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
