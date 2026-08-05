"""
This module supports benchmarking ArangoDB's experimental vector index.

The vector index is a faiss-based IVF (inverted file) index. The build-time
tunable is `nLists` (number of Voronoi cells the vectors are clustered into);
the query-time tunable is `nProbe` (how many of those cells are scanned per
query). Larger nProbe -> higher recall, lower QPS.

Optionally a faiss `factory` string can be supplied to benchmark composite
indexes (e.g. an HNSW coarse quantizer with scalar/product quantization of the
residuals: `IVF{}_HNSW32,SQ8` or `IVF{}_HNSW32,PQ64`). The `{}` placeholder is
replaced by ArangoDB with the resolved `nLists`, so the factory's implied nlist
always agrees with the `nLists` param. The factory must resolve to an IVF index
(hence the `IVF{}_` prefix). Note that product quantization (`PQ<m>`) requires
the subquantizer count `m` to divide the vector dimension, so PQ factory strings
are dataset-specific while scalar quantization (`SQ*`) works for any dimension.

Requires ArangoDB >= 3.12.4 started with `--experimental-vector-index true`.

Connection is configured via environment variables so the same module works
whether ArangoDB runs in a separate container (recommended for manual runs
with `run.py --local`) or is launched locally by this module:

    ANN_BENCHMARKS_ARANGO_HOST      default: 127.0.0.1
    ANN_BENCHMARKS_ARANGO_PORT      default: 8529
    ANN_BENCHMARKS_ARANGO_USER      default: root
    ANN_BENCHMARKS_ARANGO_PASSWORD  default: "" (empty)
    ANN_BENCHMARKS_ARANGO_DB        default: _system
    ANN_BENCHMARKS_ARANGO_TIMEOUT   default: 3600 (seconds)

Building the vector index runs synchronously (inBackground: False) and trains
the IVF quantizer, which on a full training set takes far longer than the
python-arango default read timeout of 60s. ANN_BENCHMARKS_ARANGO_TIMEOUT raises
that limit so the build request is not cut off mid-flight.

If ANN_BENCHMARKS_ARANGO_START_SERVICE is truthy (default: false), this module
launches a local single-server `arangod` itself, which requires the `arangod`
binary to be on PATH.
"""

import os
import subprocess
import sys
import time

import numpy as np
from arango import ArangoClient

from ..base.module import BaseANN
from ...util import get_bool_env_var

DATA_DIR = "/tmp/arangodb-data"
APPS_DIR = "/tmp/arangodb-apps"

METRIC_PROPERTIES = {
    # ann-benchmarks metric -> (ArangoDB metric, AQL function, SORT direction)
    "euclidean": ("l2", "APPROX_NEAR_L2", "ASC"),
    "angular": ("cosine", "APPROX_NEAR_COSINE", "DESC"),
}


def _env(name, default=None):
    value = os.getenv(f"ANN_BENCHMARKS_ARANGO_{name}", default)
    if value is None or (isinstance(value, str) and len(value.strip()) == 0):
        return default
    return value


class ArangoDB(BaseANN):
    # ArangoDB's vector index is a faiss IVF index, so this mirrors the
    # faiss-ivf wrapper: the constructor takes nLists directly and nProbe is
    # set per query. Training iterations are not exposed by faiss-ivf either,
    # so we keep a fixed default here.
    TRAINING_ITERATIONS = 25

    def __init__(self, metric, n_list, factory=None):
        if metric not in METRIC_PROPERTIES:
            raise RuntimeError(
                f"unknown metric {metric}; supported: {sorted(METRIC_PROPERTIES)}"
            )
        self._metric = metric
        self._arango_metric, self._func, self._sort_dir = METRIC_PROPERTIES[metric]
        self._n_lists = n_list
        # Optional faiss factory string for composite indexes (e.g.
        # "IVF{}_HNSW32,SQ8"). None -> the default IVF-flat index.
        self._factory = factory
        self._training_iterations = self.TRAINING_ITERATIONS
        self._n_probe = None
        self._proc = None
        self._db = None

        self._host = _env("HOST", "127.0.0.1")
        self._port = int(_env("PORT", "8529"))
        self._user = _env("USER", "root")
        self._password = _env("PASSWORD", "") or ""
        self._db_name = _env("DB", "_system")
        self._request_timeout = int(_env("TIMEOUT", "3600"))

    def _connect(self):
        url = f"http://{self._host}:{self._port}"
        client = ArangoClient(hosts=url, request_timeout=self._request_timeout)
        last_err = None
        for _ in range(120):
            try:
                db = client.db(
                    self._db_name, username=self._user, password=self._password
                )
                db.version()
                self._db = db
                print(f"connected to arangod at {url}")
                return
            except Exception as e:  # noqa: BLE001 - retry until server is ready
                last_err = e
                time.sleep(1)
        raise RuntimeError(f"arangod at {url} did not become ready: {last_err}")

    def _start_server(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(APPS_DIR, exist_ok=True)
        self._proc = subprocess.Popen(
            [
                "arangod",
                "--database.directory", DATA_DIR,
                "--javascript.app-path", APPS_DIR,
                "--server.endpoint", f"tcp://{self._host}:{self._port}",
                "--server.authentication", "false",
                "--experimental-vector-index", "true",
                "--log.level", "warning",
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

    def fit(self, X):
        if get_bool_env_var("ANN_BENCHMARKS_ARANGO_START_SERVICE", False):
            self._start_server()
        self._connect()
        db = self._db
        dim = X.shape[1]

        # The base data only depends on the dataset, not on the index
        # parameters, so across a parameter sweep it is loaded once and reused:
        # each sweep step drops and rebuilds only the vector index, never the
        # collection. Reloading millions of high-dimensional vectors per step
        # dominated the runtime otherwise. Data is (re)inserted only when the
        # collection is missing or holds a different row count than the dataset.
        if db.has_collection("items"):
            coll = db.collection("items")
        else:
            coll = db.create_collection("items")

        existing = coll.count()
        if existing == len(X):
            print("reusing %d already-loaded rows, skipping insert" % existing)
        else:
            if existing != 0:
                print(
                    "collection has %d rows but dataset has %d; reloading"
                    % (existing, len(X))
                )
                coll.truncate()
            print("inserting %d rows..." % len(X))
            sys.stdout.flush()
            insert_start = time.time()
            batch = []
            batch_size = 10000
            for i, vec in enumerate(X):
                batch.append({"_key": str(i), "vector": vec.tolist()})
                if len(batch) >= batch_size:
                    coll.insert_many(batch)
                    batch = []
            if batch:
                coll.insert_many(batch)
            print("inserted in %.3f sec" % (time.time() - insert_start))

        # A previous sweep step may have left a vector index behind; drop it so
        # the current nLists/factory takes effect. The data underneath is
        # untouched.
        for index in coll.indexes():
            if index.get("name") == "vector_idx":
                coll.delete_index(index["id"])

        # Creating the index trains the IVF quantizer, so it must run after the
        # data has been loaded. add_index passes the raw definition through to
        # POST /_api/index, which is required because the vector index type is
        # not covered by python-arango's typed helpers.
        if self._factory is None:
            print("creating vector index (nLists=%d)..." % self._n_lists)
        else:
            print(
                "creating vector index (nLists=%d, factory=%s)..."
                % (self._n_lists, self._factory)
            )
        sys.stdout.flush()
        index_start = time.time()
        params = {
            "metric": self._arango_metric,
            "dimension": dim,
            "nLists": self._n_lists,
            "trainingIterations": self._training_iterations,
        }
        if self._factory is not None:
            params["factory"] = self._factory
        coll.add_index(
            {
                "type": "vector",
                "name": "vector_idx",
                "fields": ["vector"],
                "inBackground": False,
                "params": params,
            }
        )
        print("index built in %.3f sec" % (time.time() - index_start))
        self._coll = coll

    def set_query_arguments(self, n_probe):
        self._n_probe = n_probe
        self._query = (
            "FOR d IN items "
            f"SORT {self._func}(d.vector, @qp, {{nProbe: {int(n_probe)}}}) {self._sort_dir} "
            "LIMIT @n RETURN d._key"
        )

    def query(self, v, n):
        cursor = self._db.aql.execute(
            self._query, bind_vars={"qp": v.tolist(), "n": n}
        )
        return [int(key) for key in cursor]

    def get_memory_usage(self):
        if self._db is None:
            return 0
        figures = self._coll.statistics()
        # bytes -> kilobytes, to match the units used by other modules
        return figures["documents_size"] / 1024

    def done(self):
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def __str__(self):
        factory = self._factory if self._factory is not None else "IVF,Flat"
        return (
            f"ArangoDB(metric={self._arango_metric}, factory={factory}, "
            f"nLists={self._n_lists}, nProbe={self._n_probe})"
        )
