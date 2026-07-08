# Benchmarking ArangoDB with ann-benchmarks

This directory adds ArangoDB's experimental **vector index** to ann-benchmarks.

The vector index is a faiss-based **IVF** (inverted file) index, so this wrapper
mirrors the `faiss-ivf` algorithm:

- **`nLists`** (build time) — number of Voronoi cells the vectors are clustered
  into. Swept over `[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]`.
- **`nProbe`** (query time) — how many cells are scanned per query. Swept over
  `[1, 5, 10, 50, 100, 200]`. Higher `nProbe` → higher recall, lower QPS.

Only **euclidean** (`l2`) and **angular** (`cosine`) datasets are supported.
Jaccard / hamming / dot datasets will not work with this index.

Requires ArangoDB **>= 3.12.4** started with `--vector-index=true`.

## Reference hardware

The official ann-benchmarks results (the plots in the repo `README.md`) were
produced, as of April 2025, on:

- **AWS `r6i.16xlarge`** — 64 vCPU, 512 GiB RAM, Intel Ice Lake (Xeon 8375C),
  memory-optimized (8 GiB/vCPU).
- Run with `--parallelism 31`, **hyperthreading disabled**. All benchmarks are
  single-CPU (one query thread).

The closest **GCP** equivalent is **`n2-highmem-64`** (64 vCPU, 512 GB RAM,
Intel). To reproduce the "hyperthreading disabled" setup, create the instance
with `--threads-per-core=1` (→ 32 physical cores; the reference runs used
`--parallelism 31`).

> Note: QPS is hardware-dependent. Recall-vs-QPS plots only compare fairly when
> every curve on them was produced on the **same** machine and parallelism
> setting. Do not compare numbers gathered here against the reference plots.

## Setup

This approach runs ArangoDB as its own Docker container and runs the benchmark
in-process with `run.py --local` — no `install.py`, no algorithm Dockerfile
build required.

### 1. Python environment (one time)

```bash
cd ann-benchmarks
pip install -r requirements.txt
pip install python-arango        # required because we run with --local
```

### 2. Start ArangoDB from the image (with the vector index enabled)

```bash
docker run -d --name arango-bench \
  -p 8529:8529 \
  -e ARANGO_NO_AUTH=1 \
  arangodb/enterprise-preview:devel-nightly \
  --experimental-vector-index=true
```

- `ARANGO_NO_AUTH=1` → empty root password, matching this module's defaults.
- The trailing `--experimental-vector-index=true` is appended to `arangod` by
  the image entrypoint. Without it the vector index is rejected.
- For a pinned version instead of the rolling nightly, use
  `arangodb/enterprise:3.12.10` (the vector index exists since 3.12.4).

### 3. Run the benchmark for a chosen dataset

```bash
export ANN_BENCHMARKS_ARANGO_HOST=127.0.0.1
export ANN_BENCHMARKS_ARANGO_PORT=8529

python run.py --local --algorithm arangodb --dataset sift-128-euclidean --runs 3
```

- `--dataset` takes exactly **one** dataset (auto-downloaded on first use).
  Change it per run — nothing runs "all datasets".
- Valid datasets include: `sift-128-euclidean`, `fashion-mnist-784-euclidean`,
  `gist-960-euclidean`, `mnist-784-euclidean`, `glove-25-angular`,
  `glove-100-angular`, `glove-200-angular`, `nytimes-256-angular`,
  `deep-image-96-angular`.
- One invocation sweeps every `nLists` build group across every `nProbe`.
- Useful flags: `--force` (re-run over existing results), `-k 100` (search
  top-100 instead of top-10), `--runs N` (keep best of N runs).

### Smoke test first

Before a full run, validate the connect → insert → index → query path on the
tiny CI dataset:

```bash
python run.py --local --algorithm arangodb --dataset random-xs-20-angular --runs 1
```

### 4. Plot

```bash
python plot.py --dataset sift-128-euclidean --x-scale logit --y-scale log \
  -o results/arangodb-sift-128-euclidean.png
```

Repeat steps 3–4 with a different `--dataset` for each dataset you want.

## Connection environment variables

| Variable                             | Default     | Purpose                                    |
| ------------------------------------ | ----------- | ------------------------------------------ |
| `ANN_BENCHMARKS_ARANGO_HOST`         | `127.0.0.1` | ArangoDB host                              |
| `ANN_BENCHMARKS_ARANGO_PORT`         | `8529`      | ArangoDB port                              |
| `ANN_BENCHMARKS_ARANGO_USER`         | `root`      | Username                                   |
| `ANN_BENCHMARKS_ARANGO_PASSWORD`     | `` (empty)  | Password                                   |
| `ANN_BENCHMARKS_ARANGO_DB`           | `_system`   | Database name                              |
| `ANN_BENCHMARKS_ARANGO_START_SERVICE`| `false`     | If truthy, launch a local `arangod` binary instead of connecting to an external server (requires `arangod` on `PATH`) |
