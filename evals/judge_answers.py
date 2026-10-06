"""Score saved answers through JEV via the Vercel AI SDK, without rerunning RAG."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv

from evals.reporting import report_runs, run_style

RUBRIC_PATH = Path(__file__).with_name("jev_rubric.json")
MODEL = "typesafe-ai/jev"


def load_rubric(path: Path = RUBRIC_PATH) -> dict[str, Any]:
    rubric = json.loads(path.read_text(encoding="utf-8"))
    weights = rubric["weights"]
    if any(not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("Rubric weights must be positive finite numbers")
    if not math.isclose(sum(weights.values()), 1):
        raise ValueError("Rubric weights must sum to one")
    if set(weights) != set(rubric["dimensions"]) | set(rubric["derived_labels"]):
        raise ValueError("Every weighted dimension must have a rubric definition")
    if set(rubric["fact_labels"]) != set(rubric["fact_credit"]):
        raise ValueError("Every fact label must have a credit value")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1
           for value in rubric["fact_credit"].values()):
        raise ValueError("Expected-fact credit must be between zero and one")
    for dimension in rubric["dimensions"].values():
        if not 2 <= len(dimension["levels"]) <= 10:
            raise ValueError("Score rubrics require 2 to 10 ordered levels")
    return rubric


def valid_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def check_source_links(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Check source-tag and Markdown URLs against retrieved metadata, without AI."""
    required = case.get("requires_source_links", case.get("scope") != "out_of_scope")
    if not isinstance(required, bool):
        raise ValueError("requires_source_links must be a boolean")
    metadata = result.get("link_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("link_metadata must be a dictionary")
    known = {url for url in metadata.values() if valid_url(url)}
    answer = result["answer"]
    matches = set()
    for label in re.findall(r"<link>\s*(.*?)\s*</link>", answer, flags=re.DOTALL):
        url = metadata.get(" ".join(label.split()))
        if url in known:
            matches.add(url)
    # Include one level of balanced parentheses, common in source URLs.
    for match in re.finditer(r"\[[^\]\n]+\]\((<?https?://(?:[^\s()<>]|\([^\s()<>]*\))+>?)\)", answer):
        url = match.group(1).strip("<>")
        if url in known:
            matches.add(url)
    return {"required": required, "matched_urls": sorted(matches),
            "score": float(bool(matches)) if required else None}


def build_request(case: dict[str, Any], result: dict[str, Any], rubric: dict[str, Any]) -> dict[str, Any]:
    facts = case.get("expected_answer_facts")
    if not isinstance(facts, list) or not facts or any(not isinstance(f, str) or not f.strip() for f in facts):
        raise ValueError(f"Case {case.get('id')} needs nonempty expected_answer_facts")
    if not isinstance(case.get("query"), str) or not case["query"].strip():
        raise ValueError("Each case needs its original query")
    if not isinstance(result.get("answer"), str):
        raise ValueError("Each result needs a text answer")
    context = result.get("retrieved_documents")
    if not isinstance(context, list) or any(not isinstance(text, str) for text in context):
        raise ValueError("Each result needs its retrieved context as a list of strings")
    questions = {
        name: {"type": "score", "instructions": rubric["instructions"] + "\n" + dimension["question"],
               "criteria": dimension["levels"]}
        for name, dimension in rubric["dimensions"].items()
    }
    for index, fact in enumerate(facts):
        questions[f"fact_{index}"] = {
            "type": "choice", "criteria": rubric["fact_labels"],
            "instructions": {"guidance": rubric["instructions"], "question": rubric["fact_question"],
                             "expected_fact": fact},
        }
    return {"state": {"original_query": case["query"], "scope": case.get("scope"),
                      "expected_answer_facts": facts, "retrieved_context": context,
                      "generated_answer": result["answer"]}, "questions": questions}


class JevGatewayClient:
    """Evaluate requests directly with the Vercel AI SDK for Python."""

    def judge(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not (os.getenv("AI_GATEWAY_API_KEY") or os.getenv("VERCEL_OIDC_TOKEN")):
            raise ValueError("Set AI_GATEWAY_API_KEY in .env or configure VERCEL_OIDC_TOKEN")
        return asyncio.run(self._judge(requests))

    async def _judge(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        import ai
        from ai.ops import experimental as jev

        model = ai.get_model(MODEL)
        responses = []
        for request in requests:
            questions = {
                name: (jev.ChoiceQuestion(**question) if question["type"] == "choice"
                       else jev.ScoreQuestion(**question))
                for name, question in request["questions"].items()
            }
            async with asyncio.timeout(60):
                result = await jev.evaluate(model, request["state"], questions)
            raw = result.model_dump(mode="json")
            raw["answers"] = raw.pop("value")
            responses.append(raw)
        return responses


def compose_score(case: dict[str, Any], result: dict[str, Any], response: dict[str, Any],
                  rubric: dict[str, Any]) -> dict[str, Any]:
    answers = response["answers"]
    scores = {}
    for name, dimension in rubric["dimensions"].items():
        answer = answers[name]
        raw = answer.get("score")
        maximum = len(dimension["levels"]) - 1
        if answer.get("type") != "score" or not isinstance(raw, (int, float)) or not math.isfinite(raw) or not 0 <= raw <= maximum:
            raise ValueError(f"Invalid JEV score for {name}")
        scores[name] = raw / maximum
    facts = []
    for index, fact in enumerate(case["expected_answer_facts"]):
        answer = answers[f"fact_{index}"]
        label = answer.get("choice")
        if answer.get("type") != "choice" or label not in rubric["fact_credit"]:
            raise ValueError("Invalid JEV expected-fact label")
        facts.append({"expected_fact": fact, "label": label, "credit": rubric["fact_credit"][label]})
    scores["expected_information"] = sum(fact["credit"] for fact in facts) / len(facts)
    links = check_source_links(case, result)
    scores["source_link_presence"] = links["score"]
    active_weights = {name: weight for name, weight in rubric["weights"].items() if scores[name] is not None}
    total_weight = sum(active_weights.values())
    effective_weights = {name: weight / total_weight for name, weight in active_weights.items()}
    composite = 100 * sum(scores[name] * weight for name, weight in effective_weights.items())
    return {"scores": scores, "expected_facts": facts, "source_links": links,
            "effective_weights": effective_weights, "composite_score": composite,
            "raw_jev_response": response}


def evaluate_answers(report: dict[str, Any], client: Any, rubric: dict[str, Any]) -> dict[str, Any]:
    runs = report_runs(report)
    requests = []
    for run in runs:
        if not run["cases"]:
            raise ValueError("Cannot judge an empty report")
        ids = [row["case"].get("id") for row in run["cases"]]
        if any(not isinstance(case_id, str) or not case_id.strip() for case_id in ids) or len(set(ids)) != len(ids):
            raise ValueError("Judged cases need unique nonempty string IDs")
        for row in run["cases"]:
            requests.append(build_request(row["case"], row["result"], rubric))
            check_source_links(row["case"], row["result"])
    responses = client.judge(requests)
    if not isinstance(responses, list) or len(responses) != len(requests):
        raise ValueError("JEV returned an incomplete set of evaluations")
    response_iter = iter(responses)
    evaluated = []
    for run in runs:
        cases = []
        for row in run["cases"]:
            cases.append({"id": row["case"]["id"], "query": row["case"]["query"],
                          **compose_score(row["case"], row["result"], next(response_iter), rubric)})
        means = {}
        for name in rubric["weights"]:
            values = [row["scores"][name] for row in cases if row["scores"][name] is not None]
            means[name] = sum(values) / len(values) if values else None
        counts = Counter(fact["label"] for row in cases for fact in row["expected_facts"])
        evaluated.append({"configuration": deepcopy(run["configuration"]), "cases": cases,
                          "summary": {"mean_scores": means,
                                      "mean_composite_score": sum(row["composite_score"] for row in cases) / len(cases),
                                      "expected_fact_labels": {label: counts[label] for label in rubric["fact_labels"]}}})
    rubric_hash = hashlib.sha256(json.dumps(rubric, sort_keys=True).encode()).hexdigest()
    return {"judge": {"model": MODEL, "provider": "Vercel AI SDK / AI Gateway",
                      "rubric_sha256": rubric_hash, "rubric": rubric}, "runs": evaluated}


def save_quality_report(report: dict[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "answer_quality.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rubric = report["judge"]["rubric"]
    names = list(rubric["weights"])
    labels = [rubric["dimensions"].get(name, {}).get("label", rubric["derived_labels"].get(name)) for name in names]
    runs = report["runs"]
    width = 0.8 / len(runs)

    def style(run):
        return run_style(run)

    def save(fig, filename):
        try:
            fig.tight_layout()
            fig.savefig(output_dir / filename, dpi=150)
        finally:
            plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(16 if len(runs) > 2 else 12, 11))
    for index, run in enumerate(runs):
        label, color = style(run)
        offset = -0.4 + width * (index + 0.5)
        means = run["summary"]["mean_scores"]
        axes[0].bar([position + offset for position, name in enumerate(names) if means[name] is not None],
                    [100 * means[name] for name in names if means[name] is not None], width=width, label=label, color=color)
        for position, name in enumerate(names):
            if means[name] is None:
                axes[0].text(position + offset, 5, "N/A", ha="center")
        axes[1].bar([position + offset for position in range(len(run["cases"]))],
                    [row["composite_score"] for row in run["cases"]], width=width, color=color,
                    label=f"{label}, mean {run['summary']['mean_composite_score']:.1f}")
    axes[0].set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axes[0].set_title("Answer quality by rubric dimension, higher is better")
    axes[1].set_xticks(range(len(runs[0]["cases"])), [row["id"] for row in runs[0]["cases"]], rotation=45, ha="right")
    axes[1].set_title("Weighted composite score by case")
    for ax in axes:
        ax.set_ylim(0, 130 if len(runs) > 2 else 115)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel("Score out of 100")
        ax.legend(ncol=3 if len(runs) > 2 else 1, fontsize=8)
    save(fig, "answer_quality.png")

    fact_labels = list(rubric["fact_labels"])
    fig, ax = plt.subplots(figsize=(13 if len(runs) > 2 else 9, 5))
    for index, run in enumerate(runs):
        label, color = style(run)
        offset = -0.4 + width * (index + 0.5)
        counts = run["summary"]["expected_fact_labels"]
        ax.bar([position + offset for position in range(len(fact_labels))],
               [counts[name] for name in fact_labels], width=width, label=label, color=color)
    ax.set_xticks(range(len(fact_labels)), [name.replace("_", " ") for name in fact_labels])
    ax.set_ylabel("Expected fact count")
    ax.set_title("JEV labels for expected information")
    if len(runs) > 2:
        ax.margins(y=0.3)
    ax.legend(ncol=3 if len(runs) > 2 else 1, fontsize=8)
    save(fig, "expected_fact_labels.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("evals/results/comparison_results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("evals/results"))
    parser.add_argument("--rubric", type=Path, default=RUBRIC_PATH)
    args = parser.parse_args(argv)
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    report = json.loads(args.input.read_text(encoding="utf-8"))
    judged = evaluate_answers(report, JevGatewayClient(), load_rubric(args.rubric))
    save_quality_report(judged, args.output_dir)
    print(f"Saved JEV quality scores and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
