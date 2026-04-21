# Run Monte Carlo Simulations Across 1,000+ Cores in Python

Run 1 billion Monte Carlo paths across 2,000 cloud workers at the same time, get the aggregated estimate in minutes.

## The Problem

You price an exotic option, estimate a VaR, sample from a posterior, or run a physics sim. 1 billion paths on one machine = hours or days. `multiprocessing.Pool` tops out at your laptop's core count. A real HPC cluster means Slurm, MPI, and a shared filesystem.

You want to split the sim into thousands of independent chunks, run them all at once on the cloud, and aggregate the sums and sums-of-squares for a correct mean and stddev.

## The Solution (Burla)

Split N paths into 2,000 chunks. Each chunk seeds its own RNG (`42 + chunk_id`) so results are reproducible and statistically independent. Each worker returns a small dict (`sum`, `sum_sq`, `n`). The client aggregates into a global mean and stddev.

No Slurm, no MPI, no shared filesystem.

## Example

```python
import math
import numpy as np
from burla import remote_parallel_map

TOTAL = 1_000_000_000
N_CHUNKS = 2_000
PER_CHUNK = TOTAL // N_CHUNKS

params = {"S0": 100.0, "K": 95.0, "T": 1.0, "r": 0.01, "sigma": 0.3}
tasks = [(i, PER_CHUNK, params) for i in range(N_CHUNKS)]


def run_chunk(task: tuple) -> dict:
    import numpy as np

    chunk_id, n, p = task
    rng = np.random.default_rng(seed=42 + chunk_id)

    Z = rng.standard_normal(n)
    ST = p["S0"] * np.exp((p["r"] - 0.5 * p["sigma"] ** 2) * p["T"] + p["sigma"] * np.sqrt(p["T"]) * Z)
    payoff = np.maximum(ST - p["K"], 0.0) * np.exp(-p["r"] * p["T"])

    return {
        "chunk_id": chunk_id,
        "n": n,
        "sum": float(payoff.sum()),
        "sum_sq": float((payoff ** 2).sum()),
    }


# 2,000 chunks -> 2,000 workers running 500k sims each in parallel
results = remote_parallel_map(run_chunk, tasks, func_cpu=1, func_ram=2)

total_n = sum(r["n"] for r in results)
total_sum = sum(r["sum"] for r in results)
total_sum_sq = sum(r["sum_sq"] for r in results)

mean = total_sum / total_n
var = (total_sum_sq / total_n) - mean ** 2
se = math.sqrt(var / total_n)

print(f"price estimate: {mean:.6f}  stderr: {se:.6f}  (n = {total_n:,})")
```

## Why This Is Better

**vs Ray** — Ray works, but you still set up a cluster, manage actors, and babysit the scheduler. For pleasantly parallel Monte Carlo, that's pure overhead.

**vs Dask** — Dask Bag over 2,000 partitions is fine, but you run a scheduler and worker processes. Serialization of numpy arrays back to the client is noisier than a small dict per task.

**vs Slurm / MPI** — Slurm means sbatch scripts, partitions, queue waits, and a shared filesystem. MPI forces an `MPI.COMM_WORLD` world-view on an embarrassingly parallel problem that doesn't need it.

**vs AWS Batch** — job definitions, compute environments, Docker images, cold starts in minutes. Burla starts 2,000 workers in seconds.

## How It Works

Each chunk is independent and seeded by its `chunk_id`. Burla runs `run_chunk` on 2,000 workers in parallel. Each worker returns just sums — not arrays — so the return payload is tiny. The client aggregates sums on one line.

## When To Use This

- Option pricing (Asian, barrier, autocallables), CVA, and PFE sims.
- VaR / ES estimation with millions of scenarios.
- Bayesian posterior sampling with N independent chains.
- Reliability and rare-event simulation.
- Any "run N trials, report mean/stddev/quantiles" workload.

## When NOT To Use This

- Sims where paths talk to each other (interacting particle systems, MCMC with cross-chain adaptation).
- Tiny sims (< 1 million paths) — a single core is faster once you subtract startup.
- You need to keep every path's full trajectory — that's TB of data; aggregate on the worker instead.
