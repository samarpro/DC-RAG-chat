# Answer quality evaluation

Add JEV judging to saved evaluation reports using a versioned rubric, deterministic link checks, and code-controlled composite weights. Preserve the retrieval reports and compare rewriting ON/OFF in new answer-quality plots.

1. Define atomic questions, expected-fact labels, scoring levels, and weights in `evals/jev_rubric.json`. Check that weights sum to one and every criterion has clear anchors.
2. Add a separate evaluator and CLI for saved single/comparison reports, calling JEV through the Vercel AI SDK and AI Gateway. Verify expected-fact coverage, input validation, deterministic links, normalization, and composite arithmetic with fake JEV responses.
3. Add plots for every rubric dimension, composite scores, and expected-fact labels. Generate and inspect offline example plots; run live judging when `TYPESAFE_API_KEY` is configured.
4. Document usage and score limitations. Run the affected tests and CLI checks.

Status: complete. The evaluator uses Vercel's Python `ai[vercel]==0.8.0` package directly, with typed Choice and Score questions through `ai.ops.experimental.evaluate()`. The JavaScript bridge and npm dependencies are removed. All 30 Python tests pass, including actual SDK validation with a fake provider. Live judging succeeded for all 20 saved answers across rewriting ON/OFF; real quality scores and both plots were saved and visually checked in `evals/results`.
