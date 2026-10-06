"""Offline checks: python -m unittest evals.test_rag_workflow."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from qdrant_client import models

from evals.rag_workflow import RAGResult, RAGWorkflow
from evals.run_evals import comparison_runs, main, retrieval_metrics, run_evaluation, save_report


class WorkflowTests(unittest.TestCase):
    def make_workflow(self, mode="hybrid", rewriting=True):
        llm = Mock()
        llm.models.generate_content.side_effect = (
            [SimpleNamespace(text="rewritten query"), SimpleNamespace(text="See <link>Help</link>")]
            if rewriting else [SimpleNamespace(text="See <link>Help</link>")]
        )
        dense = Mock()
        dense.embed.return_value = SimpleNamespace(embeddings=[[0.1, 0.2]])
        sparse = Mock()
        sparse.embed.return_value = [SimpleNamespace(
            indices=np.array([1, 2]), values=np.array([0.3, 0.4])
        )]
        qdrant = Mock()
        qdrant.query_points.return_value = SimpleNamespace(points=[
            SimpleNamespace(payload={"source": "help.txt", "text": "passage", "url_dict": {"Help": "https://example.com"}}),
            SimpleNamespace(payload=None),
        ])
        return RAGWorkflow(
            search_mode=mode, query_rewriting=rewriting, llm=llm,
            voyage_client=dense, sparse_embedding_model=sparse, qdrant_client=qdrant,
        )

    def test_all_search_and_rewrite_combinations(self):
        for mode in ("hybrid", "sparse", "dense"):
            for rewriting in (True, False):
                with self.subTest(mode=mode, rewriting=rewriting):
                    workflow = self.make_workflow(mode, rewriting)
                    result = workflow.answer("original query", limit=30)
                    query = "rewritten query" if rewriting else "original query"
                    self.assertEqual(result.expanded_query, query)
                    self.assertEqual(result.retrieved_documents, ["passage"])
                    self.assertEqual(result.ranked_sources, ["help.txt", None])
                    self.assertEqual(result.answer, "See [Help](https://example.com)")
                    self.assertEqual(workflow.llm.models.generate_content.call_count, 2 if rewriting else 1)
                    self.assertEqual(workflow.voyage_client.embed.call_count, int(mode != "sparse"))
                    self.assertEqual(workflow.sparse_embedding_model.embed.call_count, int(mode != "dense"))
                    if mode != "sparse":
                        self.assertEqual(workflow.voyage_client.embed.call_args.kwargs["texts"], [query])
                    if mode != "dense":
                        workflow.sparse_embedding_model.embed.assert_called_once_with(query)
                    args = workflow.qdrant_client.query_points.call_args
                    self.assertEqual(args.args, ("hybrid-search-splade",))
                    self.assertEqual(args.kwargs["limit"], 30)
                    if mode == "hybrid":
                        self.assertEqual(args.kwargs["query"].fusion, models.Fusion.RRF)
                        self.assertEqual([p.using for p in args.kwargs["prefetch"]], ["voyage3", "splade"])
                        self.assertTrue(all(p.limit >= 30 for p in args.kwargs["prefetch"]))
                    else:
                        self.assertNotIn("prefetch", args.kwargs)
                        self.assertEqual(args.kwargs["using"], "voyage3" if mode == "dense" else "splade")
                        if mode == "sparse":
                            self.assertIsInstance(args.kwargs["query"], models.SparseVector)
                        else:
                            self.assertEqual(args.kwargs["query"], [0.1, 0.2])

    def test_defaults(self):
        workflow = self.make_workflow()
        self.assertEqual(workflow.search_mode, "hybrid")
        self.assertTrue(workflow.query_rewriting)

    def test_unused_embedding_clients_are_not_initialized(self):
        for mode, forbidden in (("dense", "SparseTextEmbedding"), ("sparse", "VoyageClient")):
            with self.subTest(mode=mode), patch(f"evals.rag_workflow.{forbidden}") as factory:
                RAGWorkflow(search_mode=mode, llm=Mock(), qdrant_client=Mock(),
                            voyage_client=Mock() if mode == "dense" else None,
                            sparse_embedding_model=Mock() if mode == "sparse" else None)
                factory.assert_not_called()

    def test_invalid_configuration_fails_before_clients(self):
        with patch("evals.rag_workflow.genai.Client") as client:
            with self.assertRaises(ValueError):
                RAGWorkflow(search_mode="invalid")
            client.assert_not_called()
        workflow = self.make_workflow()
        with self.assertRaises(ValueError):
            workflow.answer("query", limit=0)
        workflow.llm.models.generate_content.assert_not_called()

    def test_wrapper_preserves_references_and_writes_graph(self):
        workflow = self.make_workflow("dense", False)
        cases = [{"id": "case-1", "query": "query", "expected_answer_facts": ["a fact"]}]
        report = run_evaluation(workflow, cases, limit=3)
        self.assertEqual(report["cases"][0]["case"], cases[0])
        self.assertEqual(report["configuration"], {
            "search_mode": "dense", "query_rewriting": False, "limit": 3, "hit_ks": [3, 5],
        })
        self.assertGreaterEqual(report["cases"][0]["latency_seconds"], 0)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            save_report(report, output)
            self.assertEqual(json.loads((output / "results.json").read_text()), report)
            self.assertNotIn("retrieved_document_count", report["cases"][0])
            for name in ("hit_at_3_and_5.png", "latency.png", "mrr.png"):
                self.assertTrue((output / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertFalse((output / "diagnostics.png").exists())

    def test_wrapper_validates_all_cases_before_running(self):
        workflow = self.make_workflow()
        with self.assertRaises(ValueError):
            run_evaluation(workflow, [{"id": "a", "query": "valid"}, {"id": "b", "query": ""}])
        workflow.llm.models.generate_content.assert_not_called()

    def test_cli_forwards_configuration(self):
        workflow = self.make_workflow("sparse", False)
        with TemporaryDirectory() as directory, patch("evals.run_evals.RAGWorkflow", return_value=workflow) as factory:
            main(["--query", "question", "--search-mode", "sparse", "--no-query-rewriting",
                  "--limit", "4", "--output-dir", directory])
            factory.assert_called_once_with(search_mode="sparse", query_rewriting=False)
            report = json.loads((Path(directory) / "results.json").read_text())
            self.assertEqual(report["configuration"]["limit"], 4)

    def test_cli_runs_both_rewriting_settings_when_requested(self):
        on = self.make_workflow("dense", True)
        off = self.make_workflow("dense", False)
        with TemporaryDirectory() as directory, patch(
            "evals.run_evals.RAGWorkflow", side_effect=[on, off]
        ) as factory:
            main(["--query", "question", "--search-mode", "dense",
                  "--compare-query-rewriting", "--output-dir", directory])
            self.assertEqual([call.kwargs["query_rewriting"] for call in factory.call_args_list], [True, False])
            comparison = json.loads((Path(directory) / "comparison_results.json").read_text())
            self.assertTrue(comparison["query_rewriting_on"]["configuration"]["query_rewriting"])
            self.assertFalse(comparison["query_rewriting_off"]["configuration"]["query_rewriting"])

    def test_comparison_aligns_cases_by_id_and_rejects_mismatches(self):
        from copy import deepcopy

        cases = [{"id": "a", "query": "first"}, {"id": "b", "query": "second"}]
        on_workflow = Mock(search_mode="dense", query_rewriting=True)
        on_workflow.answer.return_value = RAGResult("q", "q", [], {}, "answer", [])
        on = run_evaluation(on_workflow, cases)
        off = deepcopy(on)
        off["configuration"]["query_rewriting"] = False
        off["cases"].reverse()
        runs = comparison_runs(off, on)
        self.assertTrue(runs[0]["configuration"]["query_rewriting"])
        self.assertEqual([row["case"]["id"] for row in runs[1]["cases"]], ["b", "a"])
        with TemporaryDirectory() as directory:
            save_report(on, Path(directory), counterpart=off)
            comparison = json.loads((Path(directory) / "comparison_results.json").read_text())
            self.assertEqual([row["case"]["id"] for row in comparison["query_rewriting_off"]["cases"]], ["a", "b"])
        for mutation in ("search_mode", "limit", "query", "query_rewriting"):
            invalid = deepcopy(off)
            if mutation == "query":
                invalid["cases"][0]["case"]["query"] = "changed"
            else:
                invalid["configuration"][mutation] = on["configuration"][mutation] if mutation == "query_rewriting" else "different"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                comparison_runs(on, invalid)


class RetrievalMetricTests(unittest.TestCase):
    def test_any_all_and_reciprocal_rank_use_original_passage_ranks(self):
        metrics = retrieval_metrics(["a.txt", "b.txt"], ["other.txt", "a.txt", "a.txt", "b.txt"])
        self.assertEqual(metrics, {
            "status": "scored", "hit_at_k": {"3": {"any": 1, "all": 0}, "5": {"any": 1, "all": 1}},
            "first_relevant_rank": 2, "reciprocal_rank_at_limit": 0.5,
        })
        self.assertEqual(retrieval_metrics(["a.txt", "b.txt"], ["a.txt", "b.txt"])["hit_at_k"]["3"]["all"], 1)

    def test_cutoff_boundary_and_misses(self):
        for sources, hit, rr in ((["x", "x", "a"], 1, 1 / 3),
                                 (["x", "x", "x", "a"], 0, 0.25),
                                 (["x"], 0, 0.0), ([], 0, 0.0)):
            with self.subTest(sources=sources):
                metrics = retrieval_metrics(["a"], sources)
                self.assertEqual(metrics["hit_at_k"]["3"]["any"], hit)
                self.assertEqual(metrics["reciprocal_rank_at_limit"], rr)

    def test_rank_five_hits_only_at_five_and_rank_six_misses_both(self):
        for rank in (5, 6):
            with self.subTest(rank=rank):
                metrics = retrieval_metrics(["a"], ["x"] * (rank - 1) + ["a"])
                self.assertEqual(metrics["hit_at_k"]["3"], {"any": 0, "all": 0})
                self.assertEqual(metrics["hit_at_k"]["5"],
                                 {"any": int(rank == 5), "all": int(rank == 5)})
                self.assertEqual(metrics["reciprocal_rank_at_limit"], 1 / rank)

    def test_exact_basename_matching_and_duplicate_references(self):
        metrics = retrieval_metrics(["a.txt", "a.txt", "b.txt"],
                                    ["/data/a.txt", "C:\\data\\b.txt"])
        self.assertEqual(metrics["hit_at_k"]["3"]["all"], 1)
        self.assertEqual(retrieval_metrics(["a.txt"], ["A.txt"])["hit_at_k"]["3"]["any"], 0)

    def test_unlabeled_and_missing_sources_are_not_scored(self):
        self.assertEqual(retrieval_metrics([], [None])["status"], "no_references")
        metrics = retrieval_metrics(["a"], [None, "a"])
        self.assertEqual(metrics["status"], "missing_source")
        self.assertIsNone(metrics["reciprocal_rank_at_limit"])

    def test_summary_averages_queries_once_and_excludes_unscorable_cases(self):
        workflow = Mock(search_mode="dense", query_rewriting=False)
        workflow.answer.side_effect = [
            RAGResult("q", "q", [], {}, "answer", sources)
            for sources in (["x", "a"], ["x"], [None], ["x"])
        ]
        cases = [{"id": str(i), "query": "q", "reference_files": refs}
                 for i, refs in enumerate((["a"], ["a"], ["a"], []))]
        report = run_evaluation(workflow, cases)
        self.assertEqual(workflow.answer.call_count, 4)
        self.assertEqual(report["retrieval_summary"], {
            "scored_cases": 2, "excluded_cases": 2,
            "hit_at_k": {"3": {"any": 0.5, "all": 0.5}, "5": {"any": 0.5, "all": 0.5}},
            "mrr_at_limit": 0.25,
        })
        with TemporaryDirectory() as directory:
            save_report(report, Path(directory))
            self.assertTrue((Path(directory) / "mrr.png").is_file())

    def test_invalid_labels_and_cutoffs_fail_before_workflow_calls(self):
        for references, limit in ((["a"], 4), ("a", 10), ([None], 10)):
            with self.subTest(references=references, limit=limit):
                workflow = Mock()
                with self.assertRaises(ValueError):
                    run_evaluation(workflow, [{"id": "a", "query": "q", "reference_files": references}],
                                   limit=limit)
                workflow.answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
