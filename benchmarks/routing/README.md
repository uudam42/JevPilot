# benchmarks/routing

Data for the routing generalization benchmark. Design: [docs/BENCHMARK_DESIGN.md](../../docs/BENCHMARK_DESIGN.md).
Running it: [docs/EXPERIMENTS.md](../../docs/EXPERIMENTS.md).

| Path | Content |
|---|---|
| `catalog.json` | 22 core capabilities of the artificial analysis-toolkit world |
| `distractors.json` | 36 plausible, irrelevant capabilities |
| `dev/workflows.json` | 23 workflow cases: for writing baselines and prompts |
| `validation/workflows.json` | 9 cases: for checking changes |
| `eval/workflows.json` | 29 cases: reporting only; 14 unseen compositions, 12 paraphrased goals |

Decision cases are derived from these files at run time (see `dataset.py`).
Changing a file changes its digest, which every run manifest records.
**Do not tune rules, prompts or adapters on `eval`.**
