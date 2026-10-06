"""Offline answer-judge checks, with no paid API calls."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, Mock, patch
from types import SimpleNamespace

from evals.judge_answers import (
    JevGatewayClient, build_request, check_source_links, compose_score, evaluate_answers,
    load_rubric, save_quality_report,
)


class JudgeAnswerTests(unittest.TestCase):
    def setUp(self):
        self.rubric = load_rubric()
        self.case = {"id": "a", "query": "What do I do?", "scope": "relevant",
                     "expected_answer_facts": ["Use the form", "Submit before Friday"]}
        self.result = {"answer": "Use the [form](https://example.com/form).",
                       "retrieved_documents": ["Use the form. Submit before Friday."],
                       "link_metadata": {"Form": "https://example.com/form"}}

    def response(self, labels=("answered", "missing")):
        answers = {name: {"type": "score", "score": 2, "probabilities": {"0": 0, "1": 0, "2": 1}}
                   for name in self.rubric["dimensions"]}
        answers.update({f"fact_{index}": {"type": "choice", "choice": label}
                        for index, label in enumerate(labels)})
        return {"answers": answers, "response": {"modelId": "typesafe-ai/jev"}}

    def report(self, rewriting=True):
        return {"configuration": {"search_mode": "hybrid", "query_rewriting": rewriting,
                                  "limit": 10, "hit_ks": [3, 5]},
                "cases": [{"case": self.case, "result": self.result}]}

    def test_composite_arithmetic_and_fact_labels(self):
        scored = compose_score(self.case, self.result, self.response(), self.rubric)
        self.assertEqual(scored["scores"]["expected_information"], 0.5)
        self.assertAlmostEqual(scored["composite_score"], 82.5)
        self.assertEqual(scored["expected_facts"][1]["label"], "missing")
        partial = compose_score(self.case, self.result, self.response(("partly_answered", "contradicted")), self.rubric)
        self.assertEqual(partial["scores"]["expected_information"], 0.25)

    def test_links_are_deterministic_and_require_known_urls(self):
        for answer, expected in (("<link> Form </link>", 1),
                                 ("[Form](https://example.com/form)", 1),
                                 ("[Form](#)", 0),
                                 ("[Unknown](https://invented.example.com)", 0),
                                 ("Form", 0)):
            with self.subTest(answer=answer):
                result = {**self.result, "answer": answer}
                self.assertEqual(check_source_links(self.case, result)["score"], expected)
        self.assertEqual(check_source_links(self.case, {**self.result, "link_metadata": {"Form": "#"}})["score"], 0)

    def test_no_link_requirement_renormalizes_weights(self):
        case = {**self.case, "scope": "out_of_scope"}
        scored = compose_score(case, self.result, self.response(), self.rubric)
        self.assertIsNone(scored["scores"]["source_link_presence"])
        self.assertAlmostEqual(sum(scored["effective_weights"].values()), 1)
        self.assertAlmostEqual(scored["composite_score"], 100 * 0.775 / 0.95)

    def test_original_query_context_and_separate_fact_questions_are_sent(self):
        request = build_request(self.case, self.result, self.rubric)
        self.assertEqual(request["state"]["original_query"], self.case["query"])
        self.assertEqual(request["state"]["retrieved_context"], self.result["retrieved_documents"])
        self.assertEqual(request["questions"]["fact_1"]["instructions"]["expected_fact"], "Submit before Friday")
        self.assertNotIn("source_link_presence", request["questions"])

    def test_comparison_summary_and_plots(self):
        client = Mock()
        client.judge.return_value = [self.response(), self.response(("answered", "answered"))]
        judged = evaluate_answers({"query_rewriting_on": self.report(), "query_rewriting_off": self.report(False)}, client, self.rubric)
        self.assertEqual(judged["runs"][0]["summary"]["expected_fact_labels"]["missing"], 1)
        self.assertEqual(judged["runs"][1]["summary"]["mean_scores"]["expected_information"], 1)
        with TemporaryDirectory() as directory:
            save_quality_report(judged, Path(directory))
            self.assertEqual(json.loads((Path(directory) / "answer_quality.json").read_text()), judged)
            for name in ("answer_quality.png", "expected_fact_labels.png"):
                self.assertTrue((Path(directory) / name).read_bytes().startswith(b"\x89PNG"))

    def test_all_six_modes_are_judged_and_plotted_without_merging_runs(self):
        from evals.reporting import SEARCH_MODES

        runs = []
        for mode in SEARCH_MODES:
            for rewriting in (True, False):
                report = self.report(rewriting)
                report["configuration"]["search_mode"] = mode
                runs.append(report)
        client = Mock()
        client.judge.return_value = [self.response() for _ in runs]
        judged = evaluate_answers({"runs": runs}, client, self.rubric)
        self.assertEqual(len(client.judge.call_args.args[0]), 6)
        self.assertEqual([(run["configuration"]["search_mode"], run["configuration"]["query_rewriting"])
                          for run in judged["runs"]], [(mode, rewriting) for mode in SEARCH_MODES for rewriting in (True, False)])
        with TemporaryDirectory() as directory:
            save_quality_report(judged, Path(directory))
            saved = json.loads((Path(directory) / "answer_quality.json").read_text())
            self.assertEqual(len(saved["runs"]), 6)

    def test_invalid_inputs_fail_before_paid_calls(self):
        for field, value in (("expected_answer_facts", []), ("requires_source_links", "yes")):
            report = deepcopy(self.report())
            report["cases"][0]["case"][field] = value
            client = Mock()
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate_answers(report, client, self.rubric)
            client.judge.assert_not_called()

    def test_invalid_and_incomplete_responses_do_not_score(self):
        response = self.response()
        response["answers"]["clarity"]["score"] = float("nan")
        with self.assertRaises(ValueError):
            compose_score(self.case, self.result, response, self.rubric)
        client = Mock()
        client.judge.return_value = []
        with self.assertRaises(ValueError):
            evaluate_answers(self.report(), client, self.rubric)

    def test_python_sdk_preserves_answers_probabilities_and_confidence(self):
        from ai.ops import experimental as jev
        from ai.ops.items import Item

        response = self.response()
        response["answers"]["fact_0"].update({"probabilities": {"answered": 1.0, "partly_answered": 0.0,
                                                               "missing": 0.0, "contradicted": 0.0},
                                            "confidence": 1.0})
        provider = SimpleNamespace(name="gateway", evaluate=AsyncMock(
            return_value=Item(value=response["answers"], provider_metadata={"model": "jev"})))
        model = SimpleNamespace(id="typesafe-ai/jev", provider=provider)
        request = build_request(self.case, self.result, self.rubric)
        with patch("ai.get_model", return_value=model) as get_model, patch.dict(
            "os.environ", {"AI_GATEWAY_API_KEY": "offline-test-key"}
        ):
            results = JevGatewayClient().judge([request])
        get_model.assert_called_once_with("typesafe-ai/jev")
        questions = provider.evaluate.call_args.args[2]
        self.assertIsInstance(questions["fact_0"], jev.ChoiceQuestion)
        self.assertIsInstance(questions["clarity"], jev.ScoreQuestion)
        self.assertEqual(results[0]["answers"]["fact_0"]["confidence"], 1)
        self.assertEqual(results[0]["provider_metadata"], {"model": "jev"})

    def test_python_sdk_rejects_incomplete_provider_responses(self):
        from ai.ops.items import Item

        provider = SimpleNamespace(name="gateway", evaluate=AsyncMock(return_value=Item(value={})))
        model = SimpleNamespace(id="typesafe-ai/jev", provider=provider)
        with patch("ai.get_model", return_value=model), patch.dict(
            "os.environ", {"AI_GATEWAY_API_KEY": "offline-test-key"}
        ), self.assertRaisesRegex(ValueError, "answer fields must match"):
            JevGatewayClient().judge([build_request(self.case, self.result, self.rubric)])


if __name__ == "__main__":
    unittest.main()
