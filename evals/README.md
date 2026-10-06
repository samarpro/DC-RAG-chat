# Evaluation foundation

This directory contains the seed question set in `golden_questions_v1.json`, a reusable RAG workflow, and a testing wrapper. The wrapper records outputs and diagnostics; answer-quality scoring is not defined yet.

## Workflow boundary

Use `RAGWorkflow.answer(query)` from `rag_workflow.py` as the Streamlit-free end-to-end entry point. Its `RAGResult` exposes the input query, expanded query, retrieved passages, ranked source filenames, link metadata, and final answer. `ranked_sources` preserves every Qdrant result's rank, including results without text or source metadata. For stage-level inspection, call `expand_query`, `retrieve_documents`, and `generate_response` separately.

The constructor accepts the model, embedding, and retrieval clients so eval code can supply fakes or controlled fixtures. This module has no UI, caching, database writes, or CLI parser. CLI handling lives in `run_evals.py`.

## Run the wrapper

Install the project requirements, including Matplotlib, and configure the environment credentials described in the root README. From the project root:

```bash
# Default: seed dataset, hybrid search, query rewriting enabled
python -m evals.run_evals

# Run both rewriting settings and compare them in every graph
python -m evals.run_evals --compare-query-rewriting

# Run rewriting off and compare with an existing rewriting-on report
python -m evals.run_evals --no-query-rewriting --compare-report evals/results/results.json --output-dir evals/results/comparison

# Explicit sparse-only retrieval, with the original queries
python -m evals.run_evals --search-mode sparse --no-query-rewriting --output-dir evals/results/sparse

# Explicit dense-only retrieval, rewriting enabled by default
python -m evals.run_evals --search-mode dense --output-dir evals/results/dense

# One question instead of the dataset
python -m evals.run_evals --query "Who can help with my enrolment?"
```

`--dataset PATH` selects another JSON dataset with a top-level `cases` array. Each case needs a unique string `id` and a nonempty `query`. `--limit N` sets the maximum retrieved passages, defaulting to 10. Both hit@3 and hit@5 are always calculated from the same retrieval run; `limit` must be at least 5 for cases with references. `--query-rewriting` explicitly enables rewriting, and `--no-query-rewriting` disables it.

Each run saves four files in `--output-dir`, defaulting to `evals/results`:

- `results.json` contains configuration, original cases, workflow outputs, latency, per-query retrieval scores, and dataset averages.
- `hit_at_3_and_5.png` contains two graphs, one for hit@3 and one for hit@5. Each shows any/all reference-file hits per query and their dataset means.
- `latency.png` shows end-to-end latency per query.
- `mrr.png` shows one MRR value for the dataset, using the configured retrieval limit.

Retrieved-passage counts are omitted. Use separate output directories to preserve different runs; reusing a directory replaces its report files. Service errors stop the run.

`--compare-query-rewriting` runs each case once with rewriting on and once with it off. `--compare-report PATH` instead compares the current run with a saved report from the opposite setting. Comparisons require identical cases, search mode, retrieval limit, and hit cutoffs; case IDs align the plots even when report order differs. These two flags are mutually exclusive.

Comparison graphs keep the same filenames. Blue bars represent rewriting ON and orange bars rewriting OFF. Hit plots use solid bars for any reference and hatched bars for all references, with a legend for all four series. Latency plots pair the two settings for each case; MRR plots label each setting on the x-axis and show its scored case count. Missing scores are marked N/A. `comparison_results.json` preserves both complete reports, while `results.json` contains the primary run selected by `--query-rewriting` or `--no-query-rewriting`. Latency is measured once per case per setting and can vary between runs.

## Retrieval metrics

The wrapper compares each Qdrant payload's `source` with the case's `reference_files`, using exact, case-sensitive basenames. Directory prefixes are removed from Unix and Windows paths. Metrics use the original passage ranking: repeated chunks from one file occupy separate ranks. Duplicate reference filenames count once.

- `hit_at_k` contains entries `"3"` and `"5"`. Each entry has `any`, which is 1 if at least one reference file appears in the top `k` passages, and `all`, which is 1 if every reference file appears there. Otherwise each indicator is 0.
- `reciprocal_rank_at_limit` is the query's RR: `1 / first_relevant_rank`, or 0 when no reference appears in the retrieved list. A match outside the top `k` still contributes to RR if it is within `limit`.
- `retrieval_summary.mrr_at_limit` is the mean RR across scored queries. With the default retrieval limit, this is MRR@10. The summary also averages both hit indicators at each cutoff in `retrieval_summary.hit_at_k`.

Each query runs once. MRR averages across queries, so repeated retrieval of the same query is unnecessary. Repeated runs can measure variation from query rewriting, but are separate experiments.

Cases without reference files, including the seed dataset's out-of-scope cases, have null metrics and status `no_references`. If any retrieved result lacks a source filename, the labeled case has null metrics and status `missing_source`, because its ranking cannot be fully checked. Both statuses are excluded from aggregate denominators and marked N/A in plots. The summary records scored and excluded case counts; if none are scored, aggregate metrics are null. Empty retrieval for a labeled case scores zero. These metrics measure retrieval against the supplied references, not answer correctness or abstention.

## Call from Python

```python
from evals.rag_workflow import RAGWorkflow
from evals.run_evals import run_evaluation

workflow = RAGWorkflow(search_mode="dense", query_rewriting=False)
result = workflow.answer("Who can help with my enrolment?", limit=5)
report = run_evaluation(workflow, [{"id": "enrolment", "query": result.query}])
```

`search_mode` accepts `hybrid`, `sparse`, or `dense` and defaults to `hybrid`. Hybrid combines Voyage and SPLADE retrieval with RRF. Single-mode searches query their named Qdrant vector directly and skip the unused embedding client. With rewriting off, `expanded_query` equals the original query. Answer generation still uses Gemini in every mode.

Run the offline checks with `python -m unittest evals.test_rag_workflow`.

## Future case format

Keep evaluation scripts, datasets, scoring definitions, and dataset provenance in this directory. Cases should use stable IDs and include only the references needed by future metrics, such as expected answer facts, relevant document IDs, or expected abstention. Do not put credentials or student conversation data here.
