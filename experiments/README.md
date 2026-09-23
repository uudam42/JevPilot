# experiments/

Research code built on JevPilot. Experiments may import `jevpilot`, `domains`
and `integrations`. Core code must never import from here.

- `routing/`: the routing benchmark. It compares routers on identical states
  and capabilities. See [docs/BENCHMARKING.md](../docs/BENCHMARKING.md).

  ```bash
  python -m experiments.routing.benchmark --help
  ```
