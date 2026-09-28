"""Targeted probes, one per finding (results/probes.json). Each builds its own dataset from the workload vectors.

    python probes.py all                  # everything below, serially
    python probes.py search-exact         # F2: pgcontext.search() never walks the attached HNSW index
    python probes.py first-use            # F3: first HNSW query per fresh backend cancelled (~500 ms, 50k vectors)
    python probes.py ann-cost             # F1: execute_query cost is flat in k / ef_search and grows with N
    python probes.py cpu-split            # F4: raw index path - on-CPU vs off-CPU time per query vs pgvector
    python probes.py syscalls             # F4: what the backend blocks on (fsync of the mapped-generation cursor)
    python probes.py dimension            # F5: 384-d vs 1024-d - does dimension explain the gap? (no)

/proc-based probes (cpu-split, syscalls) need the bench container to share the PostgreSQL container's PID namespace
with CAP_SYS_PTRACE (docker-compose.yml does this).
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psycopg

from common import RESULTS_DIR, backend_pid, connect, pct, schedstat, setup_logging, vector_literal, versions, write_json
from data import make_vectors, project, workload

log = logging.getLogger("probes")
K = 8
N_SMALL = 1069
N_LARGE = 50000
SAMPLE_SECONDS = 4.0
SYSCALL_NAMES = {0: "read", 1: "write", 7: "poll", 17: "pread64", 18: "pwrite64", 44: "sendto", 45: "recvfrom",
                 74: "fsync", 75: "fdatasync", 202: "futex", 232: "epoll_wait", 281: "epoll_pwait"}
PG_DATA = Path("/proc/1/root/var/lib/postgresql/data")


def build(conn, schema: str, vecs: np.ndarray, with_pgvector: bool = True) -> None:
    """schema.vec (+ collection <schema>_coll with attached pgcontext_hnsw) and schema.vec_pgv (pgvector HNSW)."""
    dim = vecs.shape[1]
    coll = f"{schema}_coll"
    conn.execute(f"DO $$ BEGIN PERFORM pgcontext.drop_collection('{coll}'); "
                 "EXCEPTION WHEN undefined_object THEN NULL; END $$")
    conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.execute(f"CREATE SCHEMA {schema}")
    conn.execute(f"CREATE TABLE {schema}.vec (id text PRIMARY KEY, embedding pgcontext.vector({dim}) NOT NULL)")
    with conn.cursor() as cur:
        with cur.copy(f"COPY {schema}.vec (id, embedding) FROM STDIN") as cp:
            for i, v in enumerate(vecs):
                cp.write_row((str(i), vector_literal(v)))
    conn.execute(f"ANALYZE {schema}.vec")
    conn.execute(f"SELECT * FROM pgcontext.create_collection('{coll}', '{schema}.vec')")
    conn.execute(f"SELECT * FROM pgcontext.register_vector('{coll}', 'v', 'embedding', {dim}, 'cosine')")
    conn.execute(f"SELECT * FROM pgcontext.backfill_points('{coll}', 10000)")
    conn.execute("SET pgcontext.hnsw_build_parallel_workers = 4")
    conn.execute(f"CREATE INDEX vec_ctx_hnsw ON {schema}.vec USING pgcontext_hnsw (embedding pgcontext.vector_hnsw_cosine_ops)")
    conn.execute(f"SELECT pgcontext.attach_hnsw_index('{coll}', 'v', '{schema}.vec_ctx_hnsw')")
    if with_pgvector:
        conn.execute(f"CREATE TABLE {schema}.vec_pgv AS SELECT id, embedding::text::vector({dim}) AS pgv FROM {schema}.vec")
        conn.execute("SET max_parallel_maintenance_workers = 4")
        conn.execute(f"CREATE INDEX vec_pgv_hnsw ON {schema}.vec_pgv USING hnsw (pgv vector_cosine_ops) WITH (m = 16, ef_construction = 64)")
        conn.execute(f"ANALYZE {schema}.vec_pgv")


def drop(conn, schema: str) -> None:
    conn.execute(f"DO $$ BEGIN PERFORM pgcontext.drop_collection('{schema}_coll'); "
                 "EXCEPTION WHEN undefined_object THEN NULL; END $$")
    conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


def raw_index_sql(schema: str) -> str:
    return f"SELECT id FROM {schema}.vec ORDER BY embedding OPERATOR(pgcontext.<=>) %s::pgcontext.vector LIMIT %s"


def pgvector_sql(schema: str) -> str:
    return f"SELECT id FROM {schema}.vec_pgv ORDER BY pgv <=> %s::vector LIMIT %s"


def ann_sql(schema: str) -> str:
    return (f"SELECT source_key FROM pgcontext.execute_query('{schema}_coll', "
            "pgcontext.query_nearest('v', %s::pgcontext.vector, NULL::jsonb, %s))")


def search_sql(schema: str) -> str:
    return f"SELECT source_key FROM pgcontext.search('{schema}_coll', 'v', %s::pgcontext.vector, %s)"


STEER = ("SET enable_seqscan = off", "SET enable_bitmapscan = off")


def fresh(setup: tuple[str, ...] = ()) -> psycopg.Connection:
    c = connect("pgctx-repro-probe")
    c.execute("LOAD 'pgcontext'")
    for s in setup:
        c.execute(s)
    return c


MAX_ATTEMPTS = 6            # F3: a fresh backend can be cancelled on several consecutive HNSW calls


def call_until_ok(c, sql: str, params: tuple) -> dict[str, Any]:
    """Run a statement, retrying pgContext's QueryCanceled; report every attempt (ms, cancelled?)."""
    attempts = []
    for _ in range(MAX_ATTEMPTS):
        t0 = time.perf_counter()
        try:
            c.execute(sql, params).fetchall()
            attempts.append({"ok": True, "ms": round(1000 * (time.perf_counter() - t0), 1)})
            return {"ok": True, "cancellations": len(attempts) - 1, "attempts": attempts}
        except psycopg.errors.QueryCanceled as e:
            attempts.append({"ok": False, "ms": round(1000 * (time.perf_counter() - t0), 1), "error": str(e).splitlines()[0]})
    return {"ok": False, "cancellations": len(attempts), "attempts": attempts}


def scan_work(c) -> dict[str, Any]:
    return c.execute("SELECT * FROM pgcontext.hnsw_last_scan_work()").fetchone()


def serving(c) -> dict[str, Any]:
    return c.execute("SELECT pack_builds, pack_reuses, last_pack_bytes, mapped_attaches, shared_attaches "
                     "FROM pgcontext.hnsw_serving_stats()").fetchone()


def explain_buffers(c, sql: str, params: tuple) -> dict[str, Any]:
    lines = [r["QUERY PLAN"] for r in c.execute("EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) " + sql, params).fetchall()]
    buffers = next((l.strip() for l in lines if "Buffers:" in l), None)
    exec_ms = next((l.split(":")[1].strip() for l in lines if l.startswith("Execution Time")), None)
    return {"top_node": lines[0][:80], "buffers": buffers, "execution_time": exec_ms}


# ------------------------------------------------------------------ F2
def probe_search_exact(qlits: list[str], base: np.ndarray) -> dict[str, Any]:
    schema = "p_small"
    with connect("pgctx-repro-build") as b:
        b.execute("LOAD 'pgcontext'")
        build(b, schema, make_vectors(base, N_SMALL), with_pgvector=False)
    out: dict[str, Any] = {"n": N_SMALL}
    with fresh() as c:                                   # a fresh backend: no ANN call has happened in it yet
        c.execute(search_sql(schema), (qlits[3], K)).fetchall()
        out["search_fresh_backend"] = {"last_scan_work": scan_work(c), "serving": serving(c),
                                       "explain": explain_buffers(c, search_sql(schema), (qlits[3], K))}
        c.execute(ann_sql(schema), (qlits[3], K)).fetchall()
        out["execute_query_after"] = {"last_scan_work": scan_work(c), "serving": serving(c)}
        res = {}
        for ef in (8, 32, 256):                          # exact search is ef-independent; ANN is not
            c.execute(f"SET pgcontext.hnsw_ef_search = {ef}")
            t0 = time.perf_counter()
            ids = [r["source_key"] for r in c.execute(search_sql(schema), (qlits[5], K)).fetchall()]
            res[ef] = {"ms": round(1000 * (time.perf_counter() - t0), 2), "ids": ids}
        out["search_ef_independent"] = len({tuple(v["ids"]) for v in res.values()}) == 1
        out["search_by_ef_ms"] = {ef: v["ms"] for ef, v in res.items()}
    with connect("pgctx-repro-build") as b:
        drop(b, schema)
    log.info("search-exact: fresh-backend node_reads %s, packs %s, explain %s | ANN node_reads %s",
             out["search_fresh_backend"]["last_scan_work"]["node_reads"], out["search_fresh_backend"]["serving"]["pack_builds"],
             out["search_fresh_backend"]["explain"]["buffers"], out["execute_query_after"]["last_scan_work"]["node_reads"])
    return out


# ------------------------------------------------------------------ F3 + F1
def ensure_large(base: np.ndarray) -> str:
    schema = "p_large"
    with connect("pgctx-repro-build") as b:
        b.execute("LOAD 'pgcontext'")
        exists = b.execute("SELECT to_regclass('p_large.vec_ctx_hnsw') AS r").fetchone()["r"]
        if not exists:
            log.info("building the %d-vector probe dataset", N_LARGE)
            build(b, schema, make_vectors(base, N_LARGE), with_pgvector=False)
    return schema


def probe_first_use(qlits: list[str], base: np.ndarray) -> dict[str, Any]:
    schema = ensure_large(base)
    variants: list[tuple[str, tuple[str, ...], Callable[[], None] | None]] = [
        ("defaults (statement_timeout 0, query_timeout_ms NULL)", (), None),
        ("session statement_timeout = 1000s", ("SET statement_timeout = '1000s'",), None),
    ]
    out: dict[str, Any] = {"n": N_LARGE, "variants": {}}
    for name, setup, _ in variants:
        rows = []
        for i in range(3):                               # three fresh backends per variant
            with fresh(setup) as c:
                first = call_until_ok(c, ann_sql(schema), (qlits[i], K))
                t1 = time.perf_counter()
                c.execute(ann_sql(schema), (qlits[i + 10], K)).fetchall() if first["ok"] else None
                rows.append({"until_first_success": first,
                             "next_call_ms": round(1000 * (time.perf_counter() - t1), 1) if first["ok"] else None,
                             "statement_timeout": c.execute("SHOW statement_timeout").fetchone()["statement_timeout"]})
        out["variants"][name] = rows
        log.info("first-use [%s]: cancellations per fresh backend %s, attempt ms %s", name,
                 [r["until_first_success"]["cancellations"] for r in rows],
                 [[a["ms"] for a in r["until_first_success"]["attempts"]] for r in rows])
    with fresh() as c:
        out["collection_limits"] = c.execute(f"SELECT * FROM pgcontext.collection_limits('{schema}_coll')").fetchone()
    return out


def probe_ann_cost(qlits: list[str], base: np.ndarray) -> dict[str, Any]:
    """execute_query at N = 50k: latency vs k and ef_search (flat), scan work, buffers - and the raw index path."""
    schema = ensure_large(base)
    out: dict[str, Any] = {"n": N_LARGE, "execute_query": {}, "raw_index": {}}
    with fresh() as c:
        out["warmup"] = call_until_ok(c, ann_sql(schema), (qlits[0], K))     # absorb the F3 first-use cancellations
        for ef in (32, 256):
            c.execute(f"SET pgcontext.hnsw_ef_search = {ef}")
            for k in (1, 10, 100):
                lat = []
                for q in qlits[:20]:
                    t0 = time.perf_counter()
                    c.execute(ann_sql(schema), (q, k)).fetchall()
                    lat.append(1000 * (time.perf_counter() - t0))
                out["execute_query"][f"ef{ef}_k{k}"] = {"p50_ms": pct(lat, 50), "node_reads": scan_work(c)["node_reads"]}
        out["execute_query_explain"] = explain_buffers(c, ann_sql(schema), (qlits[1], 10))
    with fresh(STEER) as c:
        out["raw_index_warmup"] = call_until_ok(c, raw_index_sql(schema), (qlits[0], K))
        for ef in (32, 256):
            c.execute(f"SET pgcontext.hnsw_ef_search = {ef}")
            lat = []
            for q in qlits[:20]:
                t0 = time.perf_counter()
                c.execute(raw_index_sql(schema), (q, 10)).fetchall()
                lat.append(1000 * (time.perf_counter() - t0))
            out["raw_index"][f"ef{ef}_k10"] = {"p50_ms": pct(lat, 50), "node_reads": scan_work(c)["node_reads"]}
    log.info("ann-cost: execute_query %s | raw index %s", {k: v["p50_ms"] for k, v in out["execute_query"].items()},
             {k: v["p50_ms"] for k, v in out["raw_index"].items()})
    return out


# ------------------------------------------------------------------ F4
def _measure_backend(sql: str, setup: tuple[str, ...], qlits: list[str], n: int = 60) -> dict[str, Any]:
    with fresh(setup) as c, connect("pgctx-repro-meta") as meta:
        pid = backend_pid(c)
        for q in qlits[:10]:
            c.execute(sql, (q, K)).fetchall()
        io_sql = ("SELECT coalesce(sum(reads), 0) AS reads, coalesce(sum(hits), 0) AS hits FROM pg_stat_io "
                  "WHERE backend_type = 'client backend'")
        c.execute("SELECT pg_stat_force_next_flush()")
        io0 = meta.execute(io_sql).fetchone()
        run0, sw0 = schedstat(pid)
        t0 = time.perf_counter()
        for i in range(n):
            c.execute(sql, (qlits[i % len(qlits)], K)).fetchall()
        wall = (time.perf_counter() - t0) / n * 1000
        run1, sw1 = schedstat(pid)
        c.execute("SELECT pg_stat_force_next_flush()")
        time.sleep(0.6)
        io1 = meta.execute(io_sql).fetchone()
    on_cpu = (run1 - run0) / n / 1e6
    return {"wall_ms": round(wall, 3), "on_cpu_ms": round(on_cpu, 3), "off_cpu_ms": round(wall - on_cpu, 3),
            "voluntary_ctxsw_per_query": round((sw1 - sw0) / n, 2),
            "buffer_reads_per_query": round(float(io1["reads"] - io0["reads"]) / n, 2),
            "buffer_hits_per_query": round(float(io1["hits"] - io0["hits"]) / n, 1)}


def probe_cpu_split(qlits: list[str], base: np.ndarray) -> dict[str, Any]:
    schema = "p_small"
    with connect("pgctx-repro-build") as b:
        b.execute("LOAD 'pgcontext'")
        build(b, schema, make_vectors(base, N_SMALL))
    out = {"n": N_SMALL,
           "pgcontext_raw_index": _measure_backend(raw_index_sql(schema), STEER, qlits),
           "pgvector_raw_index": _measure_backend(pgvector_sql(schema), STEER + ("SET hnsw.ef_search = 40",), qlits)}
    for name, variant in (("mmap_serving_off", ("SET pgcontext.hnsw_mmap_serving = off",)),
                          ("shared_and_mmap_serving_off", ("SET pgcontext.hnsw_shared_serving = off",
                                                           "SET pgcontext.hnsw_mmap_serving = off")),
                          ("pack_on_first_use_off", ("SET pgcontext.hnsw_pack_on_first_use = off",))):
        out[f"pgcontext_raw_index_{name}"] = _measure_backend(raw_index_sql(schema), STEER + variant, qlits)
    log.info("cpu-split: %s", {k: (v["wall_ms"], v["on_cpu_ms"], v["voluntary_ctxsw_per_query"])
                                for k, v in out.items() if isinstance(v, dict)})
    return out


def probe_syscalls(qlits: list[str], base: np.ndarray) -> dict[str, Any]:
    """Sample the serving backend's state / current syscall / wchan / fsync target while it runs queries."""
    schema = "p_small"
    with connect("pgctx-repro-build") as b:
        b.execute("LOAD 'pgcontext'")
        if not b.execute("SELECT to_regclass('p_small.vec_pgv') AS r").fetchone()["r"]:
            build(b, schema, make_vectors(base, N_SMALL))
    out: dict[str, Any] = {"n": N_SMALL}
    for name, sql, setup in (("pgcontext_raw_index", raw_index_sql(schema), STEER),
                             ("pgvector_raw_index", pgvector_sql(schema), STEER + ("SET hnsw.ef_search = 40",))):
        with fresh(setup) as c:
            pid = backend_pid(c)
            for q in qlits[:10]:
                c.execute(sql, (q, K)).fetchall()
            states: collections.Counter = collections.Counter()
            fsync_targets: collections.Counter = collections.Counter()
            stop = threading.Event()

            def sample() -> None:
                while not stop.is_set():
                    try:
                        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
                        sc = Path(f"/proc/{pid}/syscall").read_text().split()
                        wchan = Path(f"/proc/{pid}/wchan").read_text().strip() or "-"
                        nr = sc[0]
                        label = "running" if nr == "running" else SYSCALL_NAMES.get(int(nr), nr) if nr.lstrip("-").isdigit() else nr
                        states[f"{state} {label} {wchan}"] += 1
                        if nr in ("74", "75"):
                            fd = int(sc[1], 16)
                            fsync_targets[os.readlink(f"/proc/{pid}/fd/{fd}").replace("/proc/1/root", "")] += 1
                    except (OSError, ValueError, IndexError):
                        pass

            th = threading.Thread(target=sample, daemon=True)
            th.start()
            t0, n = time.perf_counter(), 0
            while time.perf_counter() - t0 < SAMPLE_SECONDS:
                c.execute(sql, (qlits[n % len(qlits)], K)).fetchall()
                n += 1
            stop.set()
            th.join()
            total = sum(states.values()) or 1
            out[name] = {"queries": n, "samples": total,
                         "top_states_pct": {k: round(100 * v / total, 1) for k, v in states.most_common(6)},
                         "fsync_share_pct": round(100 * sum(v for k, v in states.items() if " fsync" in k or " fdatasync" in k) / total, 1),
                         "fsync_targets": dict(fsync_targets.most_common(5))}
            log.info("syscalls %s: fsync %.1f %% of samples, targets %s", name, out[name]["fsync_share_pct"], out[name]["fsync_targets"])
    mapped = PG_DATA / "base"
    listing = {}
    for d in mapped.glob("*/pgcontext_hnsw_mapped"):
        listing[str(d).replace("/proc/1/root", "")] = {
            "entries": sorted(p.name for p in d.iterdir())[:20],
            "pending_drops": {b.name: len(list(b.iterdir())) for b in (d / ".pending_drops").iterdir()} if (d / ".pending_drops").exists() else None}
    out["mapped_generation_dirs"] = listing
    return out


# ------------------------------------------------------------------ F5
def probe_dimension(qlits_unused: list[str], base: np.ndarray) -> dict[str, Any]:
    vecs = make_vectors(base, N_SMALL)
    out: dict[str, Any] = {"n": N_SMALL, "ef_search": 40}
    for dim, v in ((1024, vecs), (384, project(vecs, 384))):
        schema = f"p_dim{dim}"
        with connect("pgctx-repro-build") as b:
            b.execute("LOAD 'pgcontext'")
            build(b, schema, v)
        qs = [vector_literal(x) for x in v[:60]]          # the rows' own vectors as queries
        out[f"pgcontext_{dim}"] = _measure_backend(raw_index_sql(schema), STEER + ("SET pgcontext.hnsw_ef_search = 40",), qs)
        out[f"pgvector_{dim}"] = _measure_backend(pgvector_sql(schema), STEER + ("SET hnsw.ef_search = 40",), qs)
        with connect("pgctx-repro-build") as b:
            drop(b, schema)
    for dim in (384, 1024):
        a, b_ = out[f"pgcontext_{dim}"], out[f"pgvector_{dim}"]
        out[f"ratio_{dim}"] = {"wall": round(a["wall_ms"] / b_["wall_ms"], 2), "on_cpu": round(a["on_cpu_ms"] / b_["on_cpu_ms"], 2)}
    log.info("dimension: ratio 384 %s | ratio 1024 %s", out["ratio_384"], out["ratio_1024"])
    return out


PROBES = {"search-exact": probe_search_exact, "first-use": probe_first_use, "ann-cost": probe_ann_cost,
          "cpu-split": probe_cpu_split, "syscalls": probe_syscalls, "dimension": probe_dimension}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("probe", choices=[*PROBES, "all"])
    args = ap.parse_args()
    setup_logging()
    base, queries = workload()
    qlits = [vector_literal(q) for q in queries]
    existing = RESULTS_DIR / "probes.json"             # merge: re-running one probe keeps the others
    report: dict[str, Any] = json.loads(existing.read_text(encoding="utf-8")) if existing.exists() else {}
    with connect("pgctx-repro-meta") as conn:
        report["versions"] = versions(conn)
    names = list(PROBES) if args.probe == "all" else [args.probe]
    for name in names:
        log.info("probe %s ...", name)
        try:
            report[name] = PROBES[name](qlits, base)
        except Exception as e:     # recorded, never swallowed: a failing probe must not hide the others
            log.exception("probe %s failed", name)
            report[name] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0]}"}
        write_json("probes.json", report)
    with connect("pgctx-repro-build") as b:
        for schema in ("p_small", "p_large"):
            drop(b, schema)


if __name__ == "__main__":
    main()
