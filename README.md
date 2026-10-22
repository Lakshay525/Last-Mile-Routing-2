# Last-mile routing: SA vs LNS, zone-wise vs general, GNN warm starts
Deakin University project. Dataset: Amazon Last Mile Routing Challenge 2021.

## Research questions
1. Does LNS beat simulated annealing on asymmetric travel-time routes?
2. Does zone-wise routing beat routing the whole route at once?
3. Does a GNN + pointer-network warm start improve either solver?

## Metrics
- `gap_vs_driver`: route cost relative to the driver's actual route (negative = faster than driver).
- `sim_to_driver`: Kendall tau with the driver's order (1 = identical).
- `zone_sw`: number of zone changes along the route.
Minimising travel time and matching the driver are different goals; both are reported.

## Results (test routes, N=..., seed=...)
| method | gap vs driver | zone switches | similarity | secs/route |
|---|---|---|---|---|
(paste from results/*.csv)

## Reproduce
    pip install -e .[gnn,dev]
    $env:LMR_DATA_ROOT = "D:/path/to/almrrc2021"
    python scripts/run_experiment.py --n-routes 200 --out results/exp01_classical.csv
    python scripts/train_gnn.py --n-routes 500 --out results/gnn.pt
    python scripts/run_experiment.py --n-routes 200 --methods sa,lns,gnn,gnn_sa,gnn_lns --gnn results/gnn.pt --out results/exp02_gnn.csv
    pytest

## Experiment log
- (date) what was tried, what happened, what next.

## Limitations
- ...