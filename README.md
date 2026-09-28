# pgContext 0.3.0 vs pgvector 0.8.6 — Benchmarking

A self-contained reproduction of a comparison we ran while evaluating pgContext for a retrieval ("company brain")
proof of concept. Evokoa's [pgvector comparison](https://github.com/Evokoa/pgContext/blob/master/docs/benchmarks/pgvector.md)
reports pgContext **3.8–5.3x faster** than pgvector; on our host pgvector was **2.4–4.4x faster than pgContext's raw
index path and 40–185x faster than its collection API** at 1k–50k vectors. This repository lets you rerun exactly what
we ran, see the evidence behind each finding, and tell us where we are wrong.

Everything here was run twice: once in our original lab harness (`results/reference/`) and once from this repository
on a clean Docker project (`results/verify/`), with the same outcome. All data is synthetic (see `data/README.md`).

---

## 1. Findings

| # | Finding | Evidence (verification run unless noted) | Status |
|---|---|---|---|
| **F1** | **The registered-collection API (`execute_query` + `query_nearest`) has a per-call cost that grows with the collection size and does not depend on `k` or `ef_search`.** At 50k vectors a query costs 107–117 ms for k = 1 / 10 / 100 and ef 32 / 256, while the **same HNSW traversal** (identical `node_reads`: 241 at ef 32, 1,237 at ef 256) through the raw index path costs 2.9 / 3.6 ms. Per call at ef 64: 5.8 → 29.5 → 82 ms at 1k → 10k → 50k (lab), 8.3 → 34.9 → 150 ms (verify). No I/O involved (1,243 shared-buffer hits, 0 reads) | `probes.json` → `ann-cost`; `scale-vector.json` `pgcontext.ann` vs `pgcontext.index` | verified, both runs |
| **F2** | **`pgcontext.search()` never uses the attached HNSW index** (unfiltered): in a fresh backend `hnsw_last_scan_work()` = 0 node reads, `hnsw_serving_stats()` = 0 packs, EXPLAIN reads the whole collection (8,985 buffers at 1,069 vectors), latency is identical at ef 8 / 32 / 256 and linear in N (≈ 10 / 90 / 390 ms at 1k / 10k / 50k); `execute_query` on the same collection immediately does 357 node reads | `probes.json` → `search-exact`; `scale-vector.json` `pgcontext.search` (recall 1.000 at every size) | verified, both runs |
| **F3** | **The first HNSW calls in a fresh backend are cancelled** with `canceling statement due to statement timeout` after ≈ 507 ms, although `statement_timeout = 0` and the collection's `query_timeout_ms` is NULL. On a freshly built 50k collection **all 6 consecutive attempts** were cancelled in each of 3 fresh backends; with a session `statement_timeout = '1000s'` one attempt ran 4.0 s and succeeded, after which new backends attached in ≈ 185 ms. Under 8 concurrent fresh clients: 14 cancellations at 50k, 8 at 10k | `probes.json` → `first-use`; `scale-vector.json` `concurrency[*].first_use_cancellations` | verified, both runs |
| **F4** | **Every raw-index-path search calls `fsync()`**: 63–77 % of the serving backend's wall time is in `fsync` (syscall 74), blocked in `jbd2_log_wait_commit`, on `base/<db>/pgcontext_hnsw_mapped/.pending_drop_bucket_cursor` (and its directory) — while the pending-drop queue is empty. 3 voluntary context switches per query vs 1 for pgvector, 1.7 ms off-CPU vs 0.2 ms, 0 buffer reads. `hnsw_mmap_serving`, `hnsw_shared_serving`, `hnsw_pack_on_first_use` off do not remove it. Throughput of the raw path plateaus at ≈ 530–690 QPS at 4–8 clients at every size (pgvector 3,780–4,250 at 4 clients, 10k / 50k) | `probes.json` → `syscalls`, `cpu-split`; `scale-vector.json` concurrency | verified, both runs |
| **F5** | **Vector dimension is not the explanation**: 384-d vs 1024-d on the same rows, raw index path, ef 40 — pgContext's on-CPU time is 0.9–1.5x pgvector's, its wall time 2.7–5.2x (lab: 1.8–2.2x / 3.7–4.3x), the difference being off-CPU (F4); the gap is not larger at 1024-d | `probes.json` → `dimension` | verified, both runs |
| F6 | (positive) **pgContext's HNSW has better recall per `ef_search`**: at 50k, ef 256 → 0.982 vs pgvector 0.978 (verify) / 0.993 vs 0.937 (lab); and **filter-aware search keeps recall 1.000** under a 16.7 % filter where pgvector post-filtering returns 6.2–6.7 of 10 rows (recall 0.54–0.63) at ef 40 | `scale-vector.json` `recall_sweep`, `filtered` | verified |

**What F1–F4 mean for the comparison.** Evokoa's harness measures the raw index path
(`ORDER BY embedding OPERATOR(pgcontext.<=>) $1 LIMIT k` with seq/bitmap scans disabled). That path is 2.4–3.4x slower
than pgvector here (2.4–4.4x), and most of the gap is F4's per-search `fsync`. Our reading, which you can confirm or correct:

- **Platform.** The published numbers were measured on macOS (Apple M4 Pro). There `fsync()` does not flush the device
  (only `F_FULLFSYNC` does) and costs microseconds. On Linux ext4 it is a journal commit (≈ 0.7–0.9 ms per call here) —
  *hypothesis*.
- **Versions.** The published benchmark used pgContext **0.1.0** (pgvector 0.8.5). As far as we can tell from the
  release notes, the segmented / mapped serving that owns `.pending_drop_bucket_cursor` arrived in 0.3.0. pgvector
  0.8.5 → 0.8.6 has no HNSW changes, so the pgvector side is comparable. That F4 is new in 0.3.0 is a *hypothesis*; we
  have not run 0.1.0.
- **Scale.** The largest ratios in the published benchmark come from 1M-vector lanes, where pgvector degrades.
  We stopped at 50k, so the 1M regime is **not** tested here.
- **Client and vector size.** Our client is also Python 3.12 + psycopg 3.2, and the round-trip floor is 0.07 ms.
  Dimension is ruled out (F5).
- **Application API.** Applications (and the Polygres SDK) use the collection API, and F1 dominates there.

## 2. Reproduce

Requirements: Docker with Compose v2, ~8 GB free RAM, ~5 GB disk, x86-64 Linux host recommended (the /proc probes read
the PostgreSQL backend's `schedstat` / `syscall` / `smaps_rollup`, which needs a shared PID namespace + `SYS_PTRACE`,
both set in `docker-compose.yml`).

```bash
git clone https://github.com/Officiel-TinkerThink/pgcontext_v_pgvector_benchmark.git
cd pgcontext_v_pgvector_benchmark
python3 scripts/make_secret.py                  # random DB password -> secrets/pg_password.txt (gitignored)
docker compose build
docker compose up -d pg
docker compose run --rm bench                   # ~15-25 min, strictly serial; results -> results/run/
```

Pieces individually (inside the bench container, `docker compose run --rm bench sh -c "..."`):

```bash
python scale_vector.py --sizes 1069 10000 50000 --clients 1 4 8    # -> /results/scale-vector.json
python scale_vector.py --synthetic ...                              # data-free random vectors instead of the workload
python probes.py all | search-exact | first-use | ann-cost | cpu-split | syscalls | dimension   # -> /results/probes.json
```

If your Docker daemon cannot resolve relative bind mounts (e.g. Docker Engine inside WSL2 driven from Windows), set
`REPRO_DIR` to this directory's absolute path as the daemon sees it (`REPRO_DIR=/mnt/c/.../repo docker compose ...`).

`docker compose down -v` removes everything this project created.

## 3. What is measured

- **Data** — the 1,069 real e5-large (1024-d) embeddings of our synthetic corpus and 60 real question embeddings
  (`data/workload_e5_1024.npz`); 10k / 50k datasets are seeded replicas (`bench/data.py`, identical to our lab
  generator). Exact top-10 ground truth in numpy.
- **Indexes** — pgContext `pgcontext_hnsw` (`vector_hnsw_cosine_ops`, defaults m 16 / ef_construction 64, attached to the
  collection with `attach_hnsw_index`) and pgvector `hnsw` (m 16 / ef_construction 64); 4 build workers each.
- **Four query paths** — `pgcontext.index` (raw index path, planner steered like the published harness),
  `pgcontext.ann` (`execute_query` + `query_nearest`), `pgcontext.search` (`search()`), `pgvector`.
- **Per size** — build time / index size, recall@10 and p50 at ef_search 32/64/128/256, EXPLAIN plans steered and
  unsteered (unsteered, PostgreSQL chooses seq scan + sort at these sizes for both engines), a 16.7 % filter sweep,
  8-second throughput runs at 1 / 4 / 8 clients with per-backend RSS/PSS/USS and the container's cgroup growth.
- **Probes** — one per finding (F1–F5), see `bench/probes.py`.

## 4. Environment of our runs

| | |
|---|---|
| Host | Windows 11 laptop, 8 vCPU / 11.7 GiB visible to Docker; Docker Engine 29.8.1 inside **WSL2** (kernel 6.6.114), volumes on the WSL ext4 virtual disk |
| Containers | database `cpus: 4`, `mem_limit: 4g`, `shm_size: 1g`; bench `cpus: 4` — one benchmark process at a time |
| PostgreSQL | 17.10 (Debian 12); `shared_buffers` 256 MB, `work_mem` 16 MB, `maintenance_work_mem` 1 GB, `jit` off, `track_io_timing` on, `pgcontext.query_telemetry_enabled` off; everything else default (`statement_timeout` 0, `pgcontext.hnsw_ef_search` 32, mmap/shared serving on) |
| Images | `ghcr.io/evokoa/pgcontext:pg17-v0.3.0` (digest in `results/environment/image-digests.txt`) + `vector.so` from `pgvector/pgvector:0.8.6-pg17`; release binaries, unmodified |
| Client | Python 3.12.10, psycopg 3.2.13 (binary), numpy 2.5.3 |

## 5. Results

`results/verify/` — this repository's run (2026-09-28); `results/reference/` — the original lab run (2026-09-25) the
findings were first made in. Key rows (p50, single client; recall@10 in brackets):

| n | pgContext raw index, ef 64 | pgContext `execute_query`, ef 64 | pgContext `search()` | pgvector, ef 64 |
|---:|---|---|---|---|
| 1,069 | 2.4 ms (1.000) · lab 2.2 ms | 8.3 ms (1.000) · lab 5.8 ms | 11.7 ms (1.000) · lab 9.8 ms | 0.87 ms (1.000) · lab 0.64 ms |
| 10,000 | 2.8 ms (0.985) · lab 2.3 ms | 34.9 ms (0.985) · lab 29.5 ms | 91.5 ms (1.000) · lab 79.6 ms | 0.79 ms (0.983) · lab 0.94 ms |
| 50,000 | 3.6 ms (0.848) · lab 3.3 ms | 150 ms (0.848) · lab 82 ms | 387 ms (0.998) · lab 333 ms | 0.81 ms (0.805) · lab 0.75 ms |

Throughput at 4 clients, 50k vectors: pgvector **4,247 QPS**, pgContext raw index **685**, `execute_query` **30**,
`search()` **7.5**. Build at 50k: pgContext 11.1 s / 393.5 MiB, pgvector 14.3 s / 390.6 MiB.

Recall differs slightly between the two runs at 50k because the query vectors were re-embedded for this repository
(bit-level ONNX differences) and the replicated dataset consists of tight near-duplicate clusters, so near-ties in the
ground truth reorder; latency and every F1–F5 signal are unaffected.

## 6. Questions for the pgContext team

1. **F4** — Is an `fsync()` of `.pending_drop_bucket_cursor` on every raw-index search intended? Can it be skipped when
   there is nothing pending, batched, or made asynchronous? Is there a setting we missed?
2. **F1** — What does `execute_query` do per call that costs ≈ 100 ms at 50k points regardless of `k` / `ef_search`
   (point-map / visibility materialisation?), and is this expected for 0.3.0?
3. **F2** — Is `pgcontext.search()` meant to stay exact with an attached HNSW index, or should it route to it?
4. **F3** — Which timer cancels the first HNSW calls after ≈ 500 ms when `statement_timeout` is 0, and how should a
   connection-pooled service avoid it?
5. Is the raw index path with planner steering the path you recommend for latency-critical use, and would you expect
   F4 to disappear on 0.1.0 or on macOS?

## 7. Repository layout

```
docker-compose.yml          pg (PostgreSQL 17 + pgContext 0.3.0 + pgvector 0.8.6) and bench services
docker/pg/                  image: Evokoa release image + pgvector's vector.so; init SQL (extensions)
docker/bench/               Python 3.12 + psycopg + numpy; bench code and data baked in
bench/scale_vector.py       the four-path scale benchmark
bench/probes.py             one probe per finding
bench/data.py, common.py    data loading / replication, connection and /proc helpers
data/                       workload vectors (+ provenance, checksum)
results/verify/             this repository's run: scale-vector.json, probes.json, logs
results/reference/          the original lab run's outputs
scripts/                    run_all.sh (serial run), make_secret.py
provenance/                 how the workload file was produced (needs the original project)
```

Contact: via the issue tracker of this repository.
