#!/usr/bin/env bash
# Final experiments on the corrected data (missing zone IDs filled from the nearest stop). Run from the repo root.
# Settings were chosen on VALIDATION routes (results/tuning/*), never on the test routes.
#   plain LNS : i=5000, kmax=2          global SA : i=5,000,000, t0=0.5
#   zonal LNS : w=3, ii=1000, wi=1500, kmax=2        zonal SA : w=3, ii=50000, wi=300000, t0=0.5
# Needs: pip install -e ".[gnn,baselines,dev]" numba   (numba is optional but ~25-65x faster)
set -e
LNS="lns:i=5000:kmax=2"; ZL="w=3:ii=1000:wi=1500:kmax=2"; ZS="w=3:ii=50000:wi=300000:t0=0.5"; SA="sa:i=5000000:t0=0.5"
python scripts/build_ref_index.py                                   # reference routes from the TRAIN split only

# F1: test routes (207), iteration-based => deterministic, can share the CPU
S="driver nn"
for L in 0 15 30 60 120; do S="$S $LNS:l=$L"; done
S="$S $LNS:mu=5 $LNS:l=60:mu=5 $SA:l=0 $SA:l=60"
for O in ztsp greedy driver ref; do for L in 0 60; do S="$S zlns:order=$O:$ZL:l=$L"; done; done
for O in ztsp ref driver; do for L in 0 60; do S="$S zsa:order=$O:$ZS:l=$L"; done; done
for R in 30 100 300; do S="$S zlns:order=ref:$ZL:g=5000:rho=$R"; done
python scripts/run_specs.py --workers 6 --out results/final_f1_test.csv $S

# F2: OR-Tools baselines (time-limited => run on an otherwise idle machine)
python scripts/run_specs.py --workers 10 --out results/final_f2_ortools.csv ortools:t=10 ortools:t=2 ortools:t=10:l=60

# F3: literature comparison on the 1,107 High+Delivered routes (all splits; references are leave-one-out from TRAIN)
R=""; for k in 0 1 2 3 4 5 6 7 8 9; do R="$R random:k=$k"; done
python scripts/run_specs.py --n-routes 6112 --split all --subset high_delivered --workers 8 --out results/final_f3_literature.csv \
    driver nn $R lns:i=5000:kmax=2:cl=1 lns:i=15000:kmax=2:cl=1 lns:i=5000:kmax=2:cl=1:l=3000 lns:i=15000:kmax=2:cl=1:l=30000 \
    $LNS:l=60 zlns:order=ref:$ZL:l=60 zlns:order=driver:$ZL:l=60 zlns:order=ztsp:$ZL:l=60 $SA:l=60

# F4: clean timings (one process, nothing else running)
python scripts/run_specs.py --limit 20 --workers 1 --out results/final_f4_timing.csv $LNS $SA zlns:order=ztsp:$ZL zlns:order=ref:$ZL zsa:order=ztsp:$ZS ortools:t=10 ortools:t=2
