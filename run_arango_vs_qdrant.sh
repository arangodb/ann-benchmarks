#!/usr/bin/env bash
#
# ArangoDB vs Qdrant — single-query (NOT batch) benchmark + website.
#
# Batch mode is deliberately never used here: in batch mode ann-benchmarks
# assigns every query the *mean* batch latency, so all p50/p95/p99/p99.9
# percentile plots collapse to the same line and carry no information. Single
# query mode times each query individually AND enforces a single saturated CPU
# (no multi-threading), which also makes the Arango/Qdrant comparison fairer.
#
# Prerequisites (started separately, e.g. in tmux — this script does NOT start
# them):
#   * arangod >= 3.12.4 with --experimental-vector-index true, reachable per the
#     ANN_BENCHMARKS_ARANGO_* env vars (defaults: 127.0.0.1:8529).
#   * qdrant reachable on localhost:6333 / grpc 6334.
#
# Usage:
#   ./run_arango_vs_qdrant.sh                 # uses the defaults below
#   DATASET=hotpotqa-3072-angular ./run_arango_vs_qdrant.sh   # full 5.2M set
#   RUNS=5 OUTPUTDIR=~/website2 ./run_arango_vs_qdrant.sh

set -euo pipefail

# --- configuration (override via environment) --------------------------------
# The published run used the 10k-query variant. Switch to the full
# "hotpotqa-3072-angular" once a comparable point lands — note IVF is
# structurally handicapped on tiny datasets, so the 10k numbers flatter Qdrant.
DATASET="${DATASET:-hotpotqa-3072-angular-10k}"
COUNT="${COUNT:-10}"                 # k (nearest neighbours) — matches the -batch runs
RUNS="${RUNS:-3}"                    # best-of-N repetitions per parameter point
OUTPUTDIR="${OUTPUTDIR:-$HOME/website2}"
# Pure IVF-flat in ArangoDB (no HNSW coarse quantizer, no scalar quantization —
# full-precision vectors, exact distances within the probed cells). The
# "arangodb-ivf" config entry is a single nLists=4096 build + nProbe sweep,
# sized for the 5.2M corpus and structured to mirror Qdrant's single build +
# query-knob sweep. (The general "arangodb" entry uses a wide 9-way nLists
# sweep that is intractable on 5.2M — do not use it here.)
ALGORITHMS=("arangodb-ivf" "qdrant")

cd "$(dirname "$0")"

echo "=== Purging any stale BATCH results for ${DATASET} (k=${COUNT}) ==="
# Batch result dirs are suffixed "-batch"; drop them so the regenerated site can
# never show batch data again.
for algo in "${ALGORITHMS[@]}"; do
  stale="results/${DATASET}/${COUNT}/${algo}-batch"
  if [[ -d "${stale}" ]]; then
    echo "  removing ${stale}"
    rm -rf "${stale}"
  fi
done

echo "=== Running single-query benchmarks (NO --batch) ==="
for algo in "${ALGORITHMS[@]}"; do
  echo "--- ${algo} ---"
  python3 run.py \
    --local \
    --algorithm "${algo}" \
    --dataset "${DATASET}" \
    --count "${COUNT}" \
    --runs "${RUNS}"
done

echo "=== Regenerating website (single-query; --recompute clears cached metrics) ==="
mkdir -p "${OUTPUTDIR}"
python3 create_website.py --outputdir "${OUTPUTDIR}" --recompute

echo "=== Done. Site written to ${OUTPUTDIR} (open ${OUTPUTDIR}/index.html) ==="
