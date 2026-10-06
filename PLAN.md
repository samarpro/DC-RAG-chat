# Answer quality evaluation

Add JEV judging to saved evaluation reports using a versioned rubric, deterministic link checks, and code-controlled composite weights. Preserve the retrieval reports and compare rewriting ON/OFF in new answer-quality plots.

1. Define atomic questions, expected-fact labels, scoring levels, and weights in `evals/jev_rubric.json`. Check that weights sum to one and every criterion has clear anchors.
2. Add a separate evaluator and CLI for saved single/comparison reports, calling JEV through the Vercel AI SDK and AI Gateway. Verify expected-fact coverage, input validation, deterministic links, normalization, and composite arithmetic with fake JEV responses.
3. Add plots for every rubric dimension, composite scores, and expected-fact labels. Generate and inspect offline example plots; run live judging when Gateway authentication is configured.
4. Document usage and score limitations. Run the affected tests and CLI checks.

Status: complete. The evaluator uses Vercel's Python `ai[vercel]==0.8.0` package directly, with typed Choice and Score questions through `ai.ops.experimental.evaluate()`. The JavaScript bridge and npm dependencies are removed. All 30 Python tests pass, including actual SDK validation with a fake provider. Live judging succeeded for all 20 saved answers across rewriting ON/OFF; real quality scores and both plots were saved and visually checked in `evals/results`.

## Compare all retrieval configurations

1. Add a six-run report format, validate matching cases and limits, and use shared mode/rewriting labels and colors. Check alignment and duplicate rejection with offline tests.
2. Add `--compare-all` and extend every retrieval and JEV plot to compare dense, sparse, and hybrid, each with rewriting ON/OFF. Verify all six configurations are retained and plotted.
3. Collect the four missing dense/sparse runs and reuse the existing hybrid runs. Save each configuration independently, then judge all 60 saved answers with the same JEV rubric.
4. Inspect the final figures and document the repeatable two-command workflow.

Status: complete. All 33 offline tests pass. Collected all 40 dense/sparse answers and reused the existing 20 hybrid answers with matching cases and limits. Live JEV scoring succeeded for all 60 answers. All five six-configuration comparison images were generated, checked for matching case order and configuration labels, and visually inspected. Each configuration's original report is preserved in its own results subdirectory. Reproduce with `python -m evals.run_evals --compare-all` followed by `python -m evals.judge_answers`.
