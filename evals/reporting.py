"""Shared validation and plot labels for retrieval and answer comparisons."""

from typing import Any, Sequence

SEARCH_MODES = ("dense", "sparse", "hybrid")
RUN_COLORS = {
    ("dense", True): "#1f77b4", ("dense", False): "#8bbce0",
    ("sparse", True): "#228b22", ("sparse", False): "#90ce90",
    ("hybrid", True): "#8b45b0", ("hybrid", False): "#c19bd4",
}


def run_style(run: dict[str, Any]) -> tuple[str, str]:
    config = run["configuration"]
    mode, rewriting = config["search_mode"], config["query_rewriting"]
    return f"{mode.title()} / rewriting {'ON' if rewriting else 'OFF'}", RUN_COLORS[(mode, rewriting)]


def align_runs(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep identical cases in matching order, with one run per configuration."""
    if not runs:
        raise ValueError("At least one run is required")
    first = runs[0]
    reference = {row["case"]["id"]: row["case"] for row in first["cases"]}
    if not reference or len(reference) != len(first["cases"]):
        raise ValueError("Runs require unique case IDs and nonempty cases")
    configurations = set()
    aligned = []
    for run in runs:
        config = run["configuration"]
        mode, rewriting = config["search_mode"], config["query_rewriting"]
        if mode not in SEARCH_MODES or not isinstance(rewriting, bool):
            raise ValueError("Invalid search mode or query rewriting setting")
        key = mode, rewriting
        if key in configurations:
            raise ValueError("Duplicate run configuration")
        configurations.add(key)
        for field in ("limit", "hit_ks"):
            if config[field] != first["configuration"][field]:
                raise ValueError(f"Comparison requires matching {field}")
        rows = {row["case"]["id"]: row for row in run["cases"]}
        if len(rows) != len(run["cases"]) or reference != {key: row["case"] for key, row in rows.items()}:
            raise ValueError("Comparison requires identical cases")
        aligned.append({**run, "cases": [rows[key] for key in reference]})
    return sorted(aligned, key=lambda run: (
        SEARCH_MODES.index(run["configuration"]["search_mode"]),
        not run["configuration"]["query_rewriting"],
    ))


def report_runs(report: dict[str, Any]) -> list[dict[str, Any]]:
    if "runs" in report:
        return align_runs(report["runs"])
    if "cases" in report:
        return align_runs([report])
    return align_runs([report["query_rewriting_on"], report["query_rewriting_off"]])
