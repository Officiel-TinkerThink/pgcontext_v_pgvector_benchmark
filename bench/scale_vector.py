"""pgContext 0.3.0 vs pgvector 0.8.6 at 1,069 / 10,000 / 50,000 vectors - four query paths, same data, same host.

    python scale_vector.py --sizes 1069 10000 50000 --clients 1 4 8        (results/scale-vector.json)
    python scale_vector.py --synthetic ...                                  (data-free random vectors instead)

Engines (all 1024-d cosine, HNSW m 16 / ef_construction 64, 4 build workers):
  pgcontext.index   raw index path - the query shape of Evokoa's benchmark (docs/benchmarks/pgvector.md):
                    ORDER BY embedding OPERATOR(pgcontext.<=>) $1 LIMIT k, seq/bitmap scans disabled
  pgcontext.ann     registered-collection API: pgcontext.execute_query(collection, query_nearest(...))
  pgcontext.search  registered-collection API: pgcontext.search(collection, vector, query, k)
  pgvector          ORDER BY pgv <=> $1 LIMIT k on its HNSW index
Per size: build time and index size, recall@10 vs exact numpy ground truth at ef_search 32/64/128/256, p50 latency,
HNSW scan evidence (hnsw_last_scan_work / hnsw_serving_stats / EXPLAIN), a 16.7 % filter sweep, and 1/4/8-client
throughput with per-backend memory. The first HNSW call in a fresh backend can be cancelled (finding C17); warm-up
retries it and records the count and the first-call time.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import threading
import time
from typing import Any

import numpy as np
import psycopg

from common import backend_memory, backend_pid, connect, pg_memory_current_mib, setup_logging, versions, vector_literal, write_json
from data import exact_topk, make_vectors, synthetic_workload, workload

log = logging.getLogger("scale_vector")
K = 10
BUILD_WORKERS = 4
EF_SEARCH = (32, 64, 128, 256)
SECONDS = 8.0
BARRIER_TIMEOUT_S = 180
FILTER_MODULUS = 6          # cat = i % 6 -> the cat = 0 filter keeps ~16.7 % of rows
ENGINES = ("pgcontext.index", "pgcontext.ann", "pgcontext.search", "pgvector")
# Evokoa's harness disables seq and bitmap scans so the planner must pick the HNSW index; unsteered, PostgreSQL
# prefers a seq scan + sort at these sizes (the default plan is recorded as evidence).
ENGINE_SESSION = {"pgcontext.index": ("SET enable_seqscan = off", "SET enable_bitmapscan = off")}
PLAN_LINE_CHARS = 80
INDEX_PLAN_SQL = ("EXPLAIN (COSTS OFF) SELECT id FROM scalev.vec ORDER BY embedding OPERATOR(pgcontext.<=>) "
                  "%s::pgcontext.vector LIMIT %s")
WARM_CALLS = 5
WARM_RETRIES = 3
CATEGORY_FILTER = json.dumps({"must": [{"key": "cat", "match": 0}]})
SAMPLE_INTERVAL_S = 0.05

SCHEMA_SQL = """
DO $$ BEGIN PERFORM pgcontext.drop_collection('scalevec');
EXCEPTION WHEN undefined_object THEN RAISE NOTICE 'collection scalevec did not exist'; END $$;
DROP SCHEMA IF EXISTS scalev CASCADE;
CREATE SCHEMA scalev;
CREATE TABLE scalev.vec (id text PRIMARY KEY, cat integer NOT NULL, embedding pgcontext.vector(1024) NOT NULL);
"""
QUERY_SQL = {
    "pgcontext.index": "SELECT id FROM scalev.vec ORDER BY embedding OPERATOR(pgcontext.<=>) %s::pgcontext.vector LIMIT %s",
    "pgcontext.ann": "SELECT source_key AS id FROM pgcontext.execute_query('scalevec', "
                     "pgcontext.query_nearest('e5', %s::pgcontext.vector, NULL::jsonb, %s))",
    "pgcontext.search": "SELECT source_key AS id FROM pgcontext.search('scalevec', 'e5', %s::pgcontext.vector, %s)",
    "pgvector": "SELECT id FROM scalev.vec_pgv ORDER BY pgv <=> %s::vector LIMIT %s",
}


class Sampler:
    """Wall time of a step and the pg container's peak cgroup memory (page cache included) while it runs."""

    def __init__(self) -> None:
        self.peak: float | None = None
        self._stop = threading.Event()

    def __enter__(self) -> "Sampler":
        self.t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            m = pg_memory_current_mib()
            if m is not None:
                self.peak = m if self.peak is None else max(self.peak, m)
            time.sleep(SAMPLE_INTERVAL_S)

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.seconds = round(time.perf_counter() - self.t0, 3)

    def as_dict(self) -> dict[str, Any]:
        return {"seconds": self.seconds, "pg_mem_peak_mib": self.peak}


def load(conn, vecs: np.ndarray) -> dict[str, Any]:
    conn.execute(SCHEMA_SQL)
    with Sampler() as s:
        with conn.cursor() as cur:
            with cur.copy("COPY scalev.vec (id, cat, embedding) FROM STDIN") as cp:
                for i, v in enumerate(vecs):
                    cp.write_row((str(i), i % FILTER_MODULUS, vector_literal(v)))
        conn.execute("CREATE TABLE scalev.vec_pgv AS SELECT id, cat, embedding::text::vector(1024) AS pgv FROM scalev.vec")
        conn.execute("ALTER TABLE scalev.vec_pgv ADD PRIMARY KEY (id)")
        conn.execute("ANALYZE scalev.vec")
        conn.execute("ANALYZE scalev.vec_pgv")
    with Sampler() as r:
        conn.execute("SELECT * FROM pgcontext.create_collection('scalevec', 'scalev.vec')")
        conn.execute("SELECT * FROM pgcontext.register_vector('scalevec', 'e5', 'embedding', 1024, 'cosine')")
        conn.execute("SELECT * FROM pgcontext.register_filter_column('scalevec', 'cat', 'cat')")
        conn.execute("SELECT * FROM pgcontext.backfill_points('scalevec', 10000)")
    return {"copy": s.as_dict(), "register": r.as_dict(),
            "table_bytes": conn.execute("SELECT pg_total_relation_size('scalev.vec') AS b").fetchone()["b"],
            "pgv_table_bytes": conn.execute("SELECT pg_total_relation_size('scalev.vec_pgv') AS b").fetchone()["b"]}


def build_indexes(conn) -> dict[str, Any]:
    out: dict[str, Any] = {}
    conn.execute(f"SET pgcontext.hnsw_build_parallel_workers = {BUILD_WORKERS}")
    conn.execute(f"SET max_parallel_maintenance_workers = {BUILD_WORKERS}")
    with Sampler() as s:
        conn.execute("CREATE INDEX vec_ctx_hnsw ON scalev.vec USING pgcontext_hnsw (embedding pgcontext.vector_hnsw_cosine_ops)")
        conn.execute("SELECT pgcontext.attach_hnsw_index('scalevec', 'e5', 'scalev.vec_ctx_hnsw')")
    out["pgcontext"] = {"build": s.as_dict(),
                        "index_bytes": conn.execute("SELECT pg_relation_size('scalev.vec_ctx_hnsw') AS b").fetchone()["b"],
                        "estimate": {k: str(v) for k, v in conn.execute(
                            "SELECT * FROM pgcontext.estimate_index_memory('scalev.vec_ctx_hnsw')").fetchone().items()}}
    with Sampler() as s:
        conn.execute("CREATE INDEX vec_pgv_hnsw ON scalev.vec_pgv USING hnsw (pgv vector_cosine_ops) WITH (m = 16, ef_construction = 64)")
    out["pgvector"] = {"build": s.as_dict(),
                       "index_bytes": conn.execute("SELECT pg_relation_size('scalev.vec_pgv_hnsw') AS b").fetchone()["b"]}
    return out


def _set_ef(conn, engine: str, ef: int) -> None:
    conn.execute(f"SET hnsw.ef_search = {ef}" if engine == "pgvector" else f"SET pgcontext.hnsw_ef_search = {ef}")


def warm(conn, engine: str, qlits: list[str]) -> dict[str, Any]:
    """WARM_CALLS queries; a first-use cancellation (C17) is retried, counted and timed."""
    for stmt in ENGINE_SESSION.get(engine, ()):
        conn.execute(stmt)
    cancels, first_ms = 0, None
    for q in qlits[:WARM_CALLS]:
        for attempt in range(WARM_RETRIES + 1):
            t0 = time.perf_counter()
            try:
                conn.execute(QUERY_SQL[engine], (q, K)).fetchall()
                break
            except psycopg.errors.QueryCanceled as e:
                cancels += 1
                log.warning("%s warm-up call cancelled after %.0f ms (attempt %d): %s", engine,
                            1000 * (time.perf_counter() - t0), attempt + 1, str(e).splitlines()[0])
                if attempt == WARM_RETRIES:
                    raise
            finally:
                if first_ms is None:
                    first_ms = round(1000 * (time.perf_counter() - t0), 1)
    return {"first_use_cancellations": cancels, "first_call_ms": first_ms}


def _plan(conn, probe: str, steered: bool) -> list[str]:
    with conn.transaction():
        conn.execute(f"SET LOCAL enable_seqscan = {'off' if steered else 'on'}")
        conn.execute(f"SET LOCAL enable_bitmapscan = {'off' if steered else 'on'}")
        return [r["QUERY PLAN"][:PLAN_LINE_CHARS] for r in conn.execute(INDEX_PLAN_SQL, (probe, K)).fetchall()]


def scan_evidence(conn, engine: str) -> dict[str, Any] | None:
    """Which path served the last query: HNSW node reads (0 => exact), pack serving counters, plans.
    Note: search() does not reset hnsw_last_scan_work(), so after search() the counter shows the previous ANN call."""
    if engine == "pgvector":
        return None
    out = {"last_scan_work": conn.execute("SELECT * FROM pgcontext.hnsw_last_scan_work()").fetchone(),
           "serving": conn.execute("SELECT pack_builds, pack_reuses, last_pack_bytes, last_pack_millis, shared_attaches, "
                                   "shared_publishes, mapped_attaches, mapped_publishes FROM pgcontext.hnsw_serving_stats()").fetchone()}
    if engine == "pgcontext.index":
        probe = vector_literal(np.ones(1024, dtype=np.float32))
        out["plan_steered"] = _plan(conn, probe, True)
        out["plan_default"] = _plan(conn, probe, False)
        for stmt in ENGINE_SESSION[engine]:
            conn.execute(stmt)
    return out


def recall_sweep(conn, engine: str, qlits: list[str], truth: list[list[int]]) -> dict[str, Any]:
    rows = []
    warmup = warm(conn, engine, qlits)
    for ef in (EF_SEARCH if engine != "pgcontext.search" else EF_SEARCH[:1]):   # exact: ef has no effect
        _set_ef(conn, engine, ef)
        lat, rec = [], []
        for q, t in zip(qlits, truth):
            t0 = time.perf_counter()
            got = [int(r["id"]) for r in conn.execute(QUERY_SQL[engine], (q, K)).fetchall()]
            lat.append(1000 * (time.perf_counter() - t0))
            rec.append(len(set(got) & set(t)) / K)
        rows.append({"ef_search": ef, "recall@10": round(statistics.fmean(rec), 4),
                     "p50_ms": round(sorted(lat)[len(lat) // 2], 3), "mean_ms": round(statistics.fmean(lat), 3)})
        log.info("%-16s ef=%-4d recall@10 %.3f p50 %.2f ms", engine, ef, rows[-1]["recall@10"], rows[-1]["p50_ms"])
    return {"warmup": warmup, "sweep": rows, "evidence": scan_evidence(conn, engine)}


FILTERED = (
    ("pgcontext.index-postfilter", "SELECT id FROM scalev.vec WHERE cat = 0 ORDER BY embedding OPERATOR(pgcontext.<=>) "
                                   "%s::pgcontext.vector LIMIT %s",
     "SET pgcontext.hnsw_ef_search = 64; SET enable_seqscan = off; SET enable_bitmapscan = off", False),
    ("pgcontext.ann-filtered", "SELECT source_key AS id FROM pgcontext.execute_query('scalevec', pgcontext.query_nearest("
                               "'e5', %s::pgcontext.vector, %s::jsonb, %s))", "SET pgcontext.hnsw_ef_search = 64", True),
    ("pgcontext.search-filtered", "SELECT source_key AS id FROM pgcontext.search('scalevec', 'e5', %s::pgcontext.vector, %s, %s)",
     "SET pgcontext.hnsw_ef_search = 64", True),
    ("pgvector-postfilter-ef40", "SELECT id FROM scalev.vec_pgv WHERE cat = 0 ORDER BY pgv <=> %s::vector LIMIT %s",
     "SET hnsw.ef_search = 40; SET hnsw.iterative_scan = off", False),
    ("pgvector-postfilter-ef200", "SELECT id FROM scalev.vec_pgv WHERE cat = 0 ORDER BY pgv <=> %s::vector LIMIT %s",
     "SET hnsw.ef_search = 200; SET hnsw.iterative_scan = off", False),
    ("pgvector-iterative", "SELECT id FROM scalev.vec_pgv WHERE cat = 0 ORDER BY pgv <=> %s::vector LIMIT %s",
     "SET hnsw.ef_search = 40; SET hnsw.iterative_scan = relaxed_order", False),
)


def filtered_sweep(conn, qlits: list[str], vecs: np.ndarray, queries: np.ndarray) -> list[dict[str, Any]]:
    """Selective filter (cat = 0, 16.7 %). Ground truth = exact top-10 within the filtered subset."""
    ids = np.arange(len(vecs))
    keep = ids[ids % FILTER_MODULUS == 0]
    truth = [[int(keep[j]) for j in row] for row in exact_topk(vecs[keep], queries)]
    out = []
    for name, sql, setup, uses_filter_json in FILTERED:
        conn.execute(setup)
        lat, rec, got_n = [], [], []
        for q, t in zip(qlits, truth):
            params = (q, CATEGORY_FILTER, K) if uses_filter_json else (q, K)
            t0 = time.perf_counter()
            got = [int(r["id"]) for r in conn.execute(sql, params).fetchall()]
            lat.append(1000 * (time.perf_counter() - t0))
            rec.append(len(set(got) & set(t)) / K)
            got_n.append(len(got))
        out.append({"variant": name, "recall@10": round(statistics.fmean(rec), 4), "mean_returned": round(statistics.fmean(got_n), 2),
                    "p50_ms": round(sorted(lat)[len(lat) // 2], 3)})
        log.info("filtered %-26s recall@10 %.3f returned %.1f p50 %.2f ms", name, out[-1]["recall@10"],
                 out[-1]["mean_returned"], out[-1]["p50_ms"])
    conn.execute("RESET hnsw.iterative_scan; RESET enable_seqscan; RESET enable_bitmapscan")
    return out


def concurrency(engine: str, qlits: list[str], clients: int, ef: int) -> dict[str, Any]:
    base = pg_memory_current_mib() or 0.0
    conns = [connect(f"pgctx-repro-{engine}-{i}") for i in range(clients)]
    lat: list[list[float]] = [[] for _ in range(clients)]
    warmups: list[dict[str, Any] | None] = [None] * clients
    stop = threading.Event()
    barrier = threading.Barrier(clients + 1)

    def work(i: int) -> None:
        c = conns[i]
        c.execute("LOAD 'pgcontext'")
        _set_ef(c, engine, ef)
        warmups[i] = warm(c, engine, qlits)
        barrier.wait(timeout=BARRIER_TIMEOUT_S)
        j = i
        while not stop.is_set():
            t0 = time.perf_counter()
            c.execute(QUERY_SQL[engine], (qlits[j % len(qlits)], K)).fetchall()
            lat[i].append(1000 * (time.perf_counter() - t0))
            j += 1

    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(clients)]
    for t in threads:
        t.start()
    try:
        barrier.wait(timeout=BARRIER_TIMEOUT_S)
    except threading.BrokenBarrierError as e:
        raise RuntimeError(f"{engine} c={clients}: a client failed during warm-up (see thread traceback above)") from e
    time.sleep(SECONDS)
    stop.set()
    for t in threads:
        t.join(timeout=60)
    mems = [m for m in (backend_memory(backend_pid(c)) for c in conns) if m]
    serving = conns[0].execute("SELECT * FROM pgcontext.hnsw_serving_stats()").fetchone() if engine != "pgvector" else None
    after = pg_memory_current_mib() or 0.0
    for c in conns:
        c.close()
    allv = [x for l in lat for x in l]
    mean = lambda key: round(statistics.fmean(m[key] for m in mems), 1) if mems else None   # noqa: E731
    return {"engine": engine, "clients": clients, "ef_search": ef, "qps": round(len(allv) / SECONDS, 1),
            "p50_ms": round(sorted(allv)[len(allv) // 2], 3) if allv else None,
            "p95_ms": round(sorted(allv)[int(0.95 * (len(allv) - 1))], 3) if allv else None,
            "mean_rss": mean("rss"), "mean_pss": mean("pss"), "mean_uss": mean("uss"),
            "sum_pss": round(sum(m["pss"] for m in mems), 1) if mems else None,
            "pg_cgroup_delta_mib": round(after - base, 1), "serving_stats": serving,
            "first_use_cancellations": sum(w["first_use_cancellations"] for w in warmups if w),
            "first_call_ms": [w["first_call_ms"] for w in warmups if w]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", nargs="*", type=int, default=[1069, 10000, 50000])
    ap.add_argument("--clients", nargs="*", type=int, default=[1, 4, 8])
    ap.add_argument("--ef", type=int, default=64, help="ef_search used for the concurrency runs")
    ap.add_argument("--pause", type=float, default=5.0, help="seconds between concurrency runs")
    ap.add_argument("--engines", nargs="*", choices=ENGINES, default=list(ENGINES))
    ap.add_argument("--synthetic", action="store_true", help="isotropic random vectors instead of the workload file")
    args = ap.parse_args()
    setup_logging()
    base, queries = synthetic_workload() if args.synthetic else workload()
    qlits = [vector_literal(q) for q in queries]
    with connect("pgctx-repro-meta") as conn:
        report: dict[str, Any] = {"versions": versions(conn), "args": vars(args), "runs": {}}
    for n in args.sizes:
        vecs = make_vectors(base, n)
        truth = exact_topk(vecs, queries)
        with connect("pgctx-repro-sv") as conn:
            conn.execute("LOAD 'pgcontext'")
            data = load(conn, vecs)
            idx = build_indexes(conn)
            sweep = {e: recall_sweep(conn, e, qlits, truth) for e in args.engines}
            filtered = filtered_sweep(conn, qlits, vecs, queries)
        conc = []
        for e in args.engines:
            for c in args.clients:
                conc.append(concurrency(e, qlits, c, args.ef))
                log.info("n=%-6d %-16s c=%-2d qps %7.1f p50 %6.2f ms | first-use cancels %d", n, e, c, conc[-1]["qps"],
                         conc[-1]["p50_ms"] or 0, conc[-1]["first_use_cancellations"])
                time.sleep(args.pause)
        report["runs"][str(n)] = {"n": n, "raw_vector_bytes": int(vecs.nbytes), "load": data, "indexes": idx,
                                  "recall_sweep": sweep, "filtered": filtered, "concurrency": conc}
        write_json("scale-vector.json", report)
    with connect("pgctx-repro-sv") as conn:
        conn.execute(SCHEMA_SQL.split("CREATE SCHEMA")[0])


if __name__ == "__main__":
    main()
