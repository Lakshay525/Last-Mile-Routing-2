#!/usr/bin/env bash
# Seed study: the same settings as run_final.sh, repeated with different random seeds on the 207 test routes.
# Seed 0 is results/final_f1_test.csv; this script adds seeds 1-4. Run from the repo root.
set -e
LNS="lns:i=5000:kmax=2"; ZL="w=3:ii=1000:wi=1500:kmax=2"; ZS="w=3:ii=50000:wi=300000:t0=0.5"; SA="sa:i=5000000:t0=0.5"
S=""
for L in 0 60; do
  S="$S $LNS:l=$L $SA:l=$L zlns:order=ztsp:$ZL:l=$L zlns:order=ref:$ZL:l=$L zsa:order=ztsp:$ZS:l=$L zsa:order=ref:$ZS:l=$L"
done
for SEED in ${SEEDS:-1 2 3 4}; do
  python scripts/run_specs.py --workers 6 --seed $SEED --out results/final_f8_seed$SEED.csv driver $S
done
