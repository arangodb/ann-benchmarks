# Benchmarking ArangoDB with ann-benchmarks

ArangoDB's experimental vector indexes:

* **IVF** (`arangodb*`) — faiss-based inverted file index; build knob `nLists`,
  query knob `nProbe`.
* **vector-graph** (`arangodb-vector-graph`) — a Vamana/DiskANN graph over
  PQ-compressed vectors; build knobs `maxDegree` / `alpha` /
  `buildSearchListSize` / `quantization`, query knob `searchListSize` (reranking
  always on).

Run in-process with `run.py --local` against an external `arangod` (IVF: >=
3.12.4 with `--vector-index true`; vector-graph: an enterprise build shipping
the `vector-graph` index type). Angular (cosine) and euclidean (l2) datasets
only.

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

# vector-graph (Vamana/DiskANN + PQ, reranking on); PQ256x8 fits dimension 3072
python run.py --local --algorithm arangodb-vector-graph --dataset hotpotqa-3072-angular-10k --runs 3
```

### Reuse a pre-built index (skip the build)

Building the vector-graph index over a large corpus is expensive. Build it once,
then re-run the `searchListSize` sweep without rebuilding by pointing at the
already-loaded collection + index:

```bash
ANN_BENCHMARKS_ARANGO_SKIP_BUILD=1 \
  python run.py --local --algorithm arangodb-vector-graph \
    --dataset hotpotqa-3072-angular-10k --runs 3 --force
```

`fit` then skips both the insert and the index creation and fails fast if the
`items` collection or its vector index is missing. This works for either index
family.

Add `--force` to re-run over existing results. Edit `config.yml` to change the
`nLists` (build) and `nProbe` (query) sweeps.

## Plot

```bash
python plot.py --dataset sift-128-euclidean --x-scale logit --y-scale log \
  -o results/arangodb-sift-128-euclidean.png
```

## Connection env vars

`ANN_BENCHMARKS_ARANGO_HOST` (`127.0.0.1`), `_PORT` (`8529`), `_USER` (`root`),
`_PASSWORD` (empty), `_DB` (`_system`), `_TIMEOUT` (`3600`),
`_START_SERVICE` (`false`), `_SKIP_BUILD` (`false`).
