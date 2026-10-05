# Evaluation foundation

This directory contains the seed question set in `golden_questions_v1.json` and is the home for future RAG evaluation data and runners. The question set is a fixture only: no benchmark is defined or run yet.

## Workflow boundary

Use `RAGWorkflow.answer(query)` from `rag_workflow.py` as the Streamlit-free end-to-end entry point. Its `RAGResult` exposes the input query, expanded query, retrieved passages, link metadata, and final answer. For stage-level inspection, call `expand_query`, `retrieve_documents`, and `generate_response` separately.

The constructor accepts the model, embedding, and retrieval clients so future eval code can supply fakes or controlled fixtures. This module has no UI, caching, or database writes. It is also callable directly with `python -m evals.rag_workflow "your question"`.

## Future case format

Keep evaluation scripts, datasets, scoring definitions, and dataset provenance in this directory. Cases should use stable IDs and include only the references needed by future metrics, such as expected answer facts, relevant document IDs, or expected abstention. Do not put credentials or student conversation data here.
