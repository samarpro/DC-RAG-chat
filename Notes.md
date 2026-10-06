I would ship **hybrid search with query rewriting OFF**, keeping the tested retrieval limit of 10 passages. It has the highest mean JEV composite score (79.12/100) and lowest mean end-to-end latency (7.39 seconds per turn). Dense-only without rewriting is a simpler alternative, but scored slightly lower and averaged 7.93 seconds. These are provisional findings from ten queries per configuration, with hybrid timings collected separately; repeat runs are needed to establish a reliable advantage.

| Configuration | Rewriting | Mean latency/turn | JEV /100 | MRR@10 |
|---|---|---:|---:|---:|
| Dense | ON | 11.45 s | 77.27 | 0.813 |
| Dense | OFF | 7.93 s | 77.63 | 0.572 |
| Sparse | ON | 11.84 s | 77.18 | 0.658 |
| Sparse | OFF | 8.81 s | 75.43 | 0.375 |
| Hybrid | ON | 9.97 s | 75.62 | 0.667 |
| **Hybrid** | **OFF** | **7.39 s** | **79.12** | **0.529** |

Rewriting improves first-relevant-result ranking, especially for dense search, but adds a model call without improving the mean answer score for dense or hybrid. Turning it off in hybrid saves 2.58 seconds per turn (26%) and increases JEV by 3.50 points. I would prioritise useful final answers and response time over MRR alone.

For hybrid OFF, at least one reference file appears in the top five for **7/8 cases (87.5%)**; all reference files appear in both the top three and top five for **5/8 (62.5%)**. The denominator excludes two out-of-scope questions. This does not establish that generating from only five passages preserves answer quality, so keep ten until that change is evaluated.

**Cost per turn:**  An illustrative paid-tier budget using Gemini 3.8 Flash rates, 5,000 input tokens, 500 billable output tokens (including thinking), and a 30-token Voyage query is **US$0.00563 per turn**, or **US$5.63 per 1,000 turns**. A 200-input/50-output-token rewrite adds about US$0.00034, before context changes. Rates are US$0.75/$3.75 per million Gemini input/output tokens through December 2026 and US$0.06 per million Voyage-3 tokens. Hosting, Qdrant and local SPLADE compute are additional. [Google pricing](https://ai.google.dev/gemini-api/docs/pricing), [Voyage pricing](https://docs.voyageai.com/docs/pricing). The workflow uses `gemini-flash-latest`; pin the evaluated version and record usage to establish production costs.

**Out-of-scope questions:** all six configurations declined to invent gaming-PC settings or tomorrow's weather. Hybrid OFF scored 72.11 and 88.11 respectively, taking 5.43 and 9.54 seconds. Both still retrieve context and generate an answer, so abstention incurs processing costs. We should ship a concise limitation and appropriate redirect, then test broader coverage. Include these cases in answer-quality scoring, while excluding reference retrieval and source-link requirements.

**Case 9 needs review:** the broad preparation answer is useful, but the expected answer requires clarification. Its lower score is not automatically a false negative. Decide whether a grounded overview plus a clarifying question is acceptable, and update the rubric consistently. Review generic link labels resolving to unrelated pages too: link presence alone does not establish relevance.

Evidence: [retrieval and latency](evals/results/comparison_results.json), [JEV scores](evals/results/answer_quality.json).
