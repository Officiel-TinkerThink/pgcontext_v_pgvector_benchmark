# Verification run (this repository, 2026-09-28)

- `run.log` — `docker compose run --rm bench` (scale_vector.py, then probes.py all). In that first pass the
  `first-use` and `ann-cost` probes stopped on an uncaught `QueryCanceled`: at 50k vectors a fresh backend is
  cancelled on several consecutive HNSW calls (finding F3), and those probes only tolerated one. They were changed
  to retry and record every attempt (`call_until_ok` in `bench/probes.py`) and re-run on their own:
  `run-probes-first-use-ann-cost.log`.
- `probes.json` — all six probes (the two re-run probes merged in), `scale-vector.json` — the scale benchmark,
  `summary.txt` — condensed numbers side by side with the original lab run (`../reference/`).
