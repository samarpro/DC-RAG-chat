# Evaluation foundation

This directory contains the seed question set in `golden_questions_v1.json`, a reusable RAG workflow, and a testing wrapper. The wrapper records outputs and diagnostics; answer-quality scoring is not defined yet.

## Workflow boundary

Use `RAGWorkflow.answer(query)` from `rag_workflow.py` as the Streamlit-free end-to-end entry point. Its `RAGResult` exposes the input query, expanded query, retrieved passages, link metadata, and final answer. For stage-level inspection, call `expand_query`, `retrieve_documents`, and `generate_response` separately.

The constructor accepts the model, embedding, and retrieval clients so eval code can supply fakes or controlled fixtures. This module has no UI, caching, database writes, or CLI parser. CLI handling lives in `run_evals.py`.

## Run the wrapper

Install the project requirements, including Matplotlib, and configure the environment credentials described in the root README. From the project root:

```bash
# Default: seed dataset, hybrid search, query rewriting enabled
python -m evals.run_evals

# Explicit sparse-only retrieval, with the original queries
python -m evals.run_evals --search-mode sparse --no-query-rewriting --output-dir evals/results/sparse

# Explicit dense-only retrieval, rewriting enabled by default
python -m evals.run_evals --search-mode dense --output-dir evals/results/dense

# One question instead of the dataset
python -m evals.run_evals --query "Who can help with my enrolment?"
```

`--dataset PATH` selects another JSON dataset with a top-level `cases` array. Each case needs a unique string `id` and a nonempty `query`. `--limit N` sets the maximum retrieved passages, defaulting to 10. `--query-rewriting` explicitly enables rewriting, and `--no-query-rewriting` disables it.

Each run saves `results.json` and `diagnostics.png` in `--output-dir`, defaulting to `evals/results`. The JSON includes the run configuration, original case references, intermediate workflow outputs, and end-to-end latency per case. Matplotlib plots latency and retrieved-passage counts using a display-free backend. These are diagnostics, not accuracy scores. Use separate output directories to preserve different runs; reusing a directory replaces its report files. Service errors stop the run.

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
