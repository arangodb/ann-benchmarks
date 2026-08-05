# Benchmarking ArangoDB with ann-benchmarks

ArangoDB's faiss-based **IVF** vector index. Run it in-process with
`run.py --local` against an external `arangod` (>= 3.12.4, started with
`--vector-index true`). Angular (cosine) and euclidean (l2) datasets only.

## Setup (one time)

```bash
pip install -r requirements.txt
pip install python-arango
```

Start a server (any of these works):

```bash
# from a local build
arangod --database.directory /tmp/arango-bench \
  --server.authentication false --vector-index true

# or from Docker
docker run -d --name arango-bench -p 8529:8529 -e ARANGO_NO_AUTH=1 \
  arangodb/enterprise:3.12.10 --experimental-vector-index=true
```

## Run

```bash
# default IVF-flat
python run.py --local --algorithm arangodb --dataset sift-128-euclidean --runs 3

# IVF + HNSW coarse quantizer + scalar quantization (any dataset)
python run.py --local --algorithm arangodb-ivf-hnsw-sq --dataset glove-100-angular --runs 3

# IVF + HNSW + product quantization (PQ<m> must divide the dimension — pick the matching entry)
python run.py --local --algorithm arangodb-ivf-hnsw-pq-glove --dataset glove-100-angular --runs 3
python run.py --local --algorithm arangodb-ivf-hnsw-pq-sift  --dataset sift-128-euclidean --runs 3
```

Add `--force` to re-run over existing results. Edit `config.yml` to change the
`nLists` (build) and `nProbe` (query) sweeps.

## Plot

```bash
python plot.py --dataset sift-128-euclidean --x-scale logit --y-scale log \
  -o results/arangodb-sift-128-euclidean.png
```

## Connection env vars

`ANN_BENCHMARKS_ARANGO_HOST` (`127.0.0.1`), `_PORT` (`8529`), `_USER` (`root`),
`_PASSWORD` (empty), `_DB` (`_system`).
