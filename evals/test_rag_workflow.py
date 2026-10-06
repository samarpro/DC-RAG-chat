"""Offline checks: python -m unittest evals.test_rag_workflow."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from qdrant_client import models

from evals.rag_workflow import RAGWorkflow
from evals.run_evals import main, run_evaluation, save_report


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
            SimpleNamespace(payload={"text": "passage", "url_dict": {"Help": "https://example.com"}}),
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
            "search_mode": "dense", "query_rewriting": False, "limit": 3,
        })
        self.assertGreaterEqual(report["cases"][0]["latency_seconds"], 0)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            save_report(report, output)
            self.assertEqual(json.loads((output / "results.json").read_text()), report)
            self.assertTrue((output / "diagnostics.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

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


if __name__ == "__main__":
    unittest.main()
