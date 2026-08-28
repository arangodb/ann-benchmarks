"""
This module supports benchmarking ArangoDB's experimental vector indexes.

Two index families are exposed, one class each:

* ``ArangoDB`` -- the faiss-based **IVF** (inverted file) index. The build-time
  tunable is ``nLists`` (number of Voronoi cells the vectors are clustered into);
  the query-time tunable is ``nProbe`` (how many of those cells are scanned per
  query). Larger nProbe -> higher recall, lower QPS. An optional faiss
  ``factory`` string benchmarks composite indexes (e.g. ``IVF{}_HNSW32,SQ8``);
  the ``{}`` placeholder is replaced by ArangoDB with the resolved ``nLists``.

* ``ArangoDBVectorGraph`` -- the **vector-graph** index, a Vamana/DiskANN graph
  over PQ-compressed vectors. Build-time tunables are ``maxDegree`` (R),
  ``alpha``, ``buildSearchListSize`` (the build-time beam width L) and
  ``quantization`` (a ``[O]PQ<M>[x<nbits>]`` spec; ``M`` must divide the vector
  dimension). The query-time tunable is ``searchListSize`` (the search beam
  width L); larger -> higher recall, lower QPS. Reranking is always enabled:
  PQ distances are approximate, so the graph's candidates are re-scored against
  the full-precision vectors before the top-k is returned.

Both index types are queried through the same ``APPROX_NEAR_COSINE`` /
``APPROX_NEAR_L2`` AQL functions; the per-family options object
(``{nProbe: ...}`` vs ``{searchListSize: ..., rerank: true}``) selects the knobs.

Requires ArangoDB started with the vector index enabled (IVF: >= 3.12.4 with
``--experimental-vector-index true``; vector-graph: an enterprise build that
ships the ``vector-graph`` index type).

Connection is configured via environment variables so the same module works
whether ArangoDB runs in a separate container (recommended for manual runs
with `run.py --local`) or is launched locally by this module:

    ANN_BENCHMARKS_ARANGO_HOST      default: 127.0.0.1
    ANN_BENCHMARKS_ARANGO_PORT      default: 8529
    ANN_BENCHMARKS_ARANGO_USER      default: root
    ANN_BENCHMARKS_ARANGO_PASSWORD  default: "" (empty)
    ANN_BENCHMARKS_ARANGO_DB        default: _system
    ANN_BENCHMARKS_ARANGO_TIMEOUT   default: 3600 (seconds)

Building the index runs synchronously (inBackground: False) and trains the
quantizer / builds the graph, which on a full training set takes far longer than
the python-arango default read timeout of 60s. ANN_BENCHMARKS_ARANGO_TIMEOUT
raises that limit so the build request is not cut off mid-flight.

If ANN_BENCHMARKS_ARANGO_START_SERVICE is truthy (default: false), this module
launches a local single-server `arangod` itself, which requires the `arangod`
binary to be on PATH.

If ANN_BENCHMARKS_ARANGO_SKIP_BUILD is truthy (default: false), `fit` neither
inserts data nor (re)builds the index: it connects and reuses the collection and
vector index already present on the server. This makes it possible to build the
index once (expensive for the graph index over a large corpus) and then re-run
the query sweep repeatedly without paying the build cost each time. The run
fails fast if the expected collection or index is missing.
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
COLLECTION = "items"

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


class _ArangoBase(BaseANN):
    """Shared connection, data-loading and query plumbing for the ArangoDB
    vector index wrappers. Subclasses own the index-specific parts: `fit`
    (index definition) and `set_query_arguments` (the query-time options)."""

    def __init__(self, metric):
        if metric not in METRIC_PROPERTIES:
            raise RuntimeError(
                f"unknown metric {metric}; supported: {sorted(METRIC_PROPERTIES)}"
            )
        self._metric = metric
        self._arango_metric, self._func, self._sort_dir = METRIC_PROPERTIES[metric]
        self._proc = None
        self._db = None
        self._coll = None
        self._query = None

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

    @property
    def _skip_build(self):
        return get_bool_env_var("ANN_BENCHMARKS_ARANGO_SKIP_BUILD", False)

    def _prepare(self):
        """Connect (optionally launching arangod first). Shared preamble for the
        subclass `fit` implementations."""
        if get_bool_env_var("ANN_BENCHMARKS_ARANGO_START_SERVICE", False):
            self._start_server()
        self._connect()

    def _load_data(self, X):
        """Insert the base vectors into `COLLECTION`, reusing what's already
        there when it matches the dataset. The base data only depends on the
        dataset, not on the index parameters, so across a parameter sweep it is
        loaded once and reused: each step rebuilds only the index, never the
        collection. Reloading millions of high-dimensional vectors per step
        dominated the runtime otherwise. Returns the collection."""
        db = self._db
        if db.has_collection(COLLECTION):
            coll = db.collection(COLLECTION)
        else:
            coll = db.create_collection(COLLECTION)

        existing = coll.count()
        if existing == len(X):
            print("reusing %d already-loaded rows, skipping insert" % existing)
            return coll

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
        return coll

    def _reuse_prebuilt(self, index_name):
        """Skip-build path: bind to the collection and index already present on
        the server, failing fast if either is missing."""
        db = self._db
        if not db.has_collection(COLLECTION):
            raise RuntimeError(
                "ANN_BENCHMARKS_ARANGO_SKIP_BUILD is set but collection "
                f"'{COLLECTION}' does not exist; build the index first"
            )
        coll = db.collection(COLLECTION)
        if not any(idx.get("name") == index_name for idx in coll.indexes()):
            raise RuntimeError(
                "ANN_BENCHMARKS_ARANGO_SKIP_BUILD is set but index "
                f"'{index_name}' is missing on collection '{COLLECTION}'; "
                "build the index first"
            )
        print(
            "skip-build: reusing %d rows and index '%s'"
            % (coll.count(), index_name)
        )
        self._coll = coll

    def _drop_index(self, coll, index_name):
        for index in coll.indexes():
            if index.get("name") == index_name:
                coll.delete_index(index["id"])

    def query(self, v, n):
        cursor = self._db.aql.execute(
            self._query, bind_vars={"qp": v.tolist(), "n": n}
        )
        return [int(key) for key in cursor]

    def get_memory_usage(self):
        if self._db is None or self._coll is None:
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


class ArangoDB(_ArangoBase):
    # ArangoDB's vector index is a faiss IVF index, so this mirrors the
    # faiss-ivf wrapper: the constructor takes nLists directly and nProbe is
    # set per query. Training iterations are not exposed by faiss-ivf either,
    # so we keep a fixed default here.
    TRAINING_ITERATIONS = 25
    INDEX_NAME = "vector_idx"

    def __init__(self, metric, n_list, factory=None):
        super().__init__(metric)
        self._n_lists = n_list
        # Optional faiss factory string for composite indexes (e.g.
        # "IVF{}_HNSW32,SQ8"). None -> the default IVF-flat index.
        self._factory = factory
        self._training_iterations = self.TRAINING_ITERATIONS
        self._n_probe = None

    def fit(self, X):
        self._prepare()
        if self._skip_build:
            self._reuse_prebuilt(self.INDEX_NAME)
            return
        coll = self._load_data(X)
        dim = X.shape[1]

        # A previous sweep step may have left a vector index behind; drop it so
        # the current nLists/factory takes effect. The data underneath is
        # untouched.
        self._drop_index(coll, self.INDEX_NAME)

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
                "name": self.INDEX_NAME,
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

    def __str__(self):
        factory = self._factory if self._factory is not None else "IVF,Flat"
        return (
            f"ArangoDB(metric={self._arango_metric}, factory={factory}, "
            f"nLists={self._n_lists}, nProbe={self._n_probe})"
        )


class ArangoDBVectorGraph(_ArangoBase):
    """ArangoDB's `vector-graph` index: a Vamana/DiskANN graph over
    PQ-compressed vectors. The graph is built once (maxDegree, alpha,
    buildSearchListSize, quantization); the query sweep varies searchListSize.
    Reranking is always on, so PQ-approximate candidates are re-scored against
    the full-precision vectors before the top-k is returned."""

    INDEX_NAME = "vector_graph_idx"

    def __init__(
        self,
        metric,
        max_degree=64,
        alpha=1.2,
        build_search_list_size=100,
        quantization="PQ256x8",
    ):
        super().__init__(metric)
        self._max_degree = max_degree
        self._alpha = alpha
        self._build_search_list_size = build_search_list_size
        self._quantization = quantization
        self._search_list_size = None

    def fit(self, X):
        self._prepare()
        if self._skip_build:
            self._reuse_prebuilt(self.INDEX_NAME)
            return
        coll = self._load_data(X)
        dim = X.shape[1]

        # A previous sweep step may have left a graph index behind; drop it so
        # the current build parameters take effect. The data is untouched.
        self._drop_index(coll, self.INDEX_NAME)

        print(
            "creating vector-graph index "
            "(maxDegree=%d, alpha=%.3f, buildSearchListSize=%d, quantization=%s)..."
            % (
                self._max_degree,
                self._alpha,
                self._build_search_list_size,
                self._quantization,
            )
        )
        sys.stdout.flush()
        index_start = time.time()
        coll.add_index(
            {
                "type": "vector-graph",
                "name": self.INDEX_NAME,
                "fields": ["vector"],
                "inBackground": False,
                "params": {
                    "metric": self._arango_metric,
                    "dimension": dim,
                    "maxDegree": self._max_degree,
                    "alpha": self._alpha,
                    "buildSearchListSize": self._build_search_list_size,
                    "quantization": self._quantization,
                },
            }
        )
        print("index built in %.3f sec" % (time.time() - index_start))
        self._coll = coll

    def set_query_arguments(self, search_list_size):
        self._search_list_size = search_list_size
        # rerank is always on: PQ distances are approximate, so the graph's
        # candidates are re-scored against the full-precision vectors.
        self._query = (
            "FOR d IN items "
            f"SORT {self._func}(d.vector, @qp, "
            f"{{searchListSize: {int(search_list_size)}, rerank: true}}) "
            f"{self._sort_dir} "
            "LIMIT @n RETURN d._key"
        )

    def __str__(self):
        return (
            f"ArangoDBVectorGraph(metric={self._arango_metric}, "
            f"maxDegree={self._max_degree}, alpha={self._alpha}, "
            f"buildSearchListSize={self._build_search_list_size}, "
            f"quantization={self._quantization}, "
            f"searchListSize={self._search_list_size}, rerank=True)"
        )
