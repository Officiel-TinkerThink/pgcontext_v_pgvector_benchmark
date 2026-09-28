"""Shared helpers: connection (password from a Docker secret file, never logged), vector literals, per-backend
process accounting through /proc (the bench container shares the PostgreSQL container's PID namespace)."""
from __future__ import annotations

import glob
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import psycopg
from psycopg.rows import dict_row

log = logging.getLogger("bench")
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results"))
CONNECT_RETRIES = 20
FLOAT32_DIGITS = 9                 # round-trips float32 exactly through the text vector format
PG_PROC_ROOT = Path("/proc/1/root")  # PID 1 of the shared namespace is the postgres postmaster


def setup_logging() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _password() -> str | None:
    path = os.environ.get("PGPASSWORD_FILE")
    return Path(path).read_text(encoding="utf-8").strip() if path else os.environ.get("PGPASSWORD")


def connect(application_name: str = "pgctx-repro", autocommit: bool = True) -> psycopg.Connection:
    """Connect with bounded retry (the database may still be starting)."""
    params = {"host": os.environ.get("PGHOST", "pg"), "port": os.environ.get("PGPORT", "5432"),
              "user": os.environ.get("PGUSER", "postgres"), "dbname": os.environ.get("PGDATABASE", "repro"),
              "application_name": application_name}
    pw = _password()
    if pw:
        params["password"] = pw
    last: Exception | None = None
    for attempt in range(1, CONNECT_RETRIES + 1):
        try:
            return psycopg.connect(**params, autocommit=autocommit, row_factory=dict_row)
        except psycopg.OperationalError as e:
            last = e
            log.warning("connect attempt %d/%d failed: %s", attempt, CONNECT_RETRIES, str(e).splitlines()[0])
            time.sleep(min(5.0, 0.5 * attempt))
    raise RuntimeError("could not connect to PostgreSQL") from last


def vector_literal(v: Sequence[float] | np.ndarray) -> str:
    return "[" + ",".join(format(float(x), f".{FLOAT32_DIGITS}g") for x in v) + "]"


def backend_pid(conn: psycopg.Connection) -> int:
    return int(conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"])


def backend_memory(pid: int) -> dict[str, float] | None:
    """RSS / PSS / USS (MiB) of one backend from smaps_rollup; None when /proc of the pg container is not visible."""
    try:
        text = Path(f"/proc/{pid}/smaps_rollup").read_text()
    except OSError:
        return None
    kb = {}
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            kb[parts[0].rstrip(":")] = int(parts[1])
    uss = kb.get("Private_Clean", 0) + kb.get("Private_Dirty", 0)
    return {"rss": round(kb.get("Rss", 0) / 1024, 1), "pss": round(kb.get("Pss", 0) / 1024, 1), "uss": round(uss / 1024, 1)}


def pg_memory_current_mib() -> float | None:
    """The pg container's cgroup v2 memory.current (includes page cache), seen through the shared PID namespace."""
    for path in (PG_PROC_ROOT / "sys/fs/cgroup/memory.current",):
        try:
            return round(int(path.read_text().strip()) / 2**20, 1)
        except (OSError, ValueError):
            continue
    return None


def schedstat(pid: int) -> tuple[int, int]:
    """(ns on CPU over all threads, voluntary context switches) of a backend."""
    run = sum(int(open(t).read().split()[0]) for t in glob.glob(f"/proc/{pid}/task/*/schedstat"))
    status = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines() if ":" in line)
    return run, int(status["voluntary_ctxt_switches"])


def pct(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))], 3)


def write_json(name: str, payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    log.info("wrote %s", path)
    return path


def versions(conn: psycopg.Connection) -> dict[str, Any]:
    conn.execute("LOAD 'pgcontext'")                   # registers the pgcontext.* settings in this session
    ext = {r["extname"]: r["extversion"] for r in conn.execute(
        "SELECT extname, extversion FROM pg_extension WHERE extname IN ('pgcontext', 'vector')").fetchall()}
    settings = {r["name"]: r["setting"] for r in conn.execute(
        "SELECT name, setting FROM pg_settings WHERE name IN ('shared_buffers', 'work_mem', 'maintenance_work_mem', "
        "'max_parallel_maintenance_workers', 'jit', 'statement_timeout', 'pgcontext.hnsw_ef_search', "
        "'pgcontext.hnsw_mmap_serving', 'pgcontext.hnsw_shared_serving', 'pgcontext.hnsw_pack_on_first_use', "
        "'hnsw.ef_search')").fetchall()}
    return {"server": conn.execute("SELECT version() AS v").fetchone()["v"], "extensions": ext, "settings": settings}
