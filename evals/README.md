# Evaluation foundation

This directory contains the seed question set in `golden_questions_v1.json`, a reusable RAG workflow, retrieval diagnostics, and a JEV answer-quality evaluator.

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

# Compare dense-only, sparse-only, and hybrid, each with rewriting ON/OFF
python -m evals.run_evals --compare-all
python -m evals.judge_answers

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

Comparison graphs keep the same filenames. Run labels include both search mode and rewriting setting. Latency, MRR, and JEV plots use blue for dense, green for sparse, and purple for hybrid, with darker shades for rewriting ON and lighter shades for OFF. In a two-run comparison, hit plots use solid bars for any reference and hatched bars for all references. Latency plots group settings for each case; MRR plots label each setting on the x-axis and show its scored case count. Missing scores are marked N/A. `comparison_results.json` preserves both complete reports, while `results.json` contains the primary run selected by `--query-rewriting` or `--no-query-rewriting`. Latency is measured once per case per setting and can vary between runs.

`--compare-all` runs all six configurations against identical cases and the same retrieval limit. It is mutually exclusive with the other comparison flags; the individual search-mode and rewriting flags apply only to single-mode runs. Results for each configuration are saved under folders such as `dense_rewriting_on` and `sparse_rewriting_off`. Root `results.json` and `comparison_results.json` contain a `runs` array with all six reports. The comparison rejects duplicate configurations, mismatched cases, limits, or hit cutoffs.

For this six-run comparison, `hit_at_3_and_5.png` shows any/all dataset hit rates by configuration, with one panel per cutoff. `latency.png` groups all six configurations by query, and `mrr.png` shows their dataset MRR values. `judge_answers` accepts the same `runs` format and compares all six configurations in the rubric, composite, and expected-fact label plots. Run each case once per configuration; these are single-run comparisons rather than estimates of timing or model variability.

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

## JEV answer quality

Judge the saved answers against their original queries, `expected_answer_facts`, and the retrieved context. This step uses JEV through the Vercel AI SDK and AI Gateway, model `typesafe-ai/jev`. It does not rerun retrieval or answer generation. The Python `ai` package supplies the Vercel SDK; no Node runtime is required.

```bash
python -m pip install -r requirements.txt
# Configure AI_GATEWAY_API_KEY in the root .env, or supply VERCEL_OIDC_TOKEN.
python -m evals.judge_answers

# Judge a single saved run instead of the ON/OFF comparison
python -m evals.judge_answers --input evals/results/results.json --output-dir evals/results/quality
```

The default input is `evals/results/comparison_results.json`. The evaluator calls `ai.ops.experimental.evaluate()` with `ai.get_model("typesafe-ai/jev")`, typed `ChoiceQuestion` and `ScoreQuestion` objects. Python validates the saved inputs, computes the composite, and draws the Matplotlib plots. SDK and provider failures stop the run rather than inventing scores. Configure Gateway credentials, not a direct TypeSafe key.

The versioned prompt and weights live in [`jev_rubric.json`](jev_rubric.json). Each semantic dimension has its own question and concrete ordered levels. JEV evaluates all questions against the same state in one request per answer. Every score is normalized to 0–1 before code applies these weights:

| Dimension | Weight | What it checks |
| --- | ---: | --- |
| Expected information | 35% | Whether the answer conveys each expected fact or expected behavior in response to the original query |
| Context grounding | 20% | Whether factual claims are supported by the context actually retrieved |
| Intent handling | 10% | Whether the answer addresses the query, clarifies ambiguity, or handles an out-of-scope request |
| Clarity | 10% | Whether the student can understand the response on first reading |
| Actionability | 10% | Whether the student has a usable answer, next step, clarification, or redirect |
| Low predicted frustration | 10% | Whether the response avoids evasiveness, dismissiveness, repetition, and unnecessary work |
| Source link presence | 5% | Deterministic presence of a source tag or Markdown link resolving to a retrieved metadata URL |

Each expected fact gets one of four JEV labels: `answered`, `partly_answered`, `missing`, or `contradicted`. Expected-information coverage averages their credits, respectively 1, 0.5, 0, and 0. An answer can be correctly grounded yet miss the reference target if retrieval did not supply the needed evidence. The other five semantic dimensions use three anchored levels, normalized by dividing JEV's score by 2. The composite is `100 * sum(effective_weight * normalized_score)`.

Link presence is checked in code against `link_metadata`. Unresolved `#` links, unknown URLs, and plain link text receive no credit. A known URL establishes that a source link is present; it does not prove that the linked page supports a particular claim or remains reachable. No external link requests are made. Cases can set `requires_source_links` explicitly; by default it is false for `out_of_scope` cases and true otherwise. When links are not required, that dimension is N/A and the remaining weights are normalized to sum to one. Other dimensions still assess out-of-scope answers against their expected abstention behavior.

The evaluator saves three new files without replacing retrieval diagnostics:

- `answer_quality.json` contains every per-fact label, normalized dimension score, effective weight, composite, raw JEV answer and probability data, model metadata, and the full rubric plus its hash.
- `answer_quality.png` plots every rubric dimension's mean and composite scores by case, comparing rewriting ON/OFF when both are supplied. Higher values are better, including low predicted frustration.
- `expected_fact_labels.png` plots all four expected-fact labels for each rewriting setting.

These scores are model judgments, not verified truth or measured user emotions. Fact-label counts count facts, while composite means count queries equally. Provider probabilities and confidence are preserved for inspection; they are not added to the quality score. Human spot checks are needed to assess whether this rubric and the seed references match the intended behavior. Change weights with `--rubric PATH` to use another version of the rubric.

The implementation follows [TypeSafe's composite scoring pattern](https://docs.typesafe.ai/patterns/composite-scoring) and [Vercel's Python evaluation API](https://ai-python.dev/docs/reference/ops#experimentalevaluate). The experimental Python SDK is pinned in `requirements.txt`.

Run offline checks with `python -m unittest evals.test_judge_answers evals.test_rag_workflow`.

## Future case format

Keep evaluation scripts, datasets, scoring definitions, and dataset provenance in this directory. Cases should use stable IDs and include only the references needed by future metrics, such as expected answer facts, relevant document IDs, or expected abstention. Do not put credentials or student conversation data here.
