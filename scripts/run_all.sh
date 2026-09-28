#!/bin/sh
# Serial run of everything (inside the bench container): scale benchmark, then the probes. ~20-30 min at 4 vCPU.
set -eu
cd /bench
echo "[$(date -u +%H:%M:%S)] scale_vector"
python scale_vector.py --sizes ${SIZES:-1069 10000 50000} --clients ${CLIENTS:-1 4 8}
echo "[$(date -u +%H:%M:%S)] probes"
python probes.py all
echo "[$(date -u +%H:%M:%S)] done - results in /results (./results/run on the host)"
