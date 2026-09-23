# experiments/

Research code built on JevPilot. Experiments may import `jevpilot`, `domains`
and `integrations`. Core code must never import from here.

- `routing/benchmark.py`: the routing **generalization** benchmark (offline
  and live). See [docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md).
- `routing/generalization/`: its world, oracle, dataset, baselines, runner and metrics.
- `routing/smoke.py`: the Phase 1 smoke benchmark. See [docs/BENCHMARKING.md](../docs/BENCHMARKING.md).

  ```bash
  python -m experiments.routing.benchmark --help
  python -m experiments.routing.smoke --help
  ```
