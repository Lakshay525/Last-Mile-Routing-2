# Last-mile routing: LNS, SA and zone-based methods on the Amazon Last Mile Routing Challenge data

Deakin University research project. We compare ways of ordering the ~150 delivery stops of a van route, judged against what the real driver did on the same route
(Amazon Last Mile Routing Research Challenge 2021, 6,112 training routes and 3,052 evaluation routes with travel-time matrices, zone labels and the driver's order).

## The three approaches (and the baseline)
1. **Approach 1, zonal LNS:** choose the order of the delivery zones, solve each zone, then re-optimise overlapping windows of three zones with large neighbourhood search (LNS).
2. **Approach 2, plain LNS:** one LNS over the whole route.
3. **Approach 3, zonal SA:** approach 1 with simulated annealing (SA) as the optimiser. Global SA is reported too.
4. **Baseline:** Google OR-Tools (routing solver; CP-SAT and MIP studied separately).

Two optional "dials": a penalty λ (seconds) for every change of zone, and a penalty ρ for going back to an earlier zone of a given zone order.
The zone order can come from simple rules, from the best-matching training route ("reference route"), or from a graph neural network (GNN).

## Headline results (207 held-out test routes, one random seed)
Travel time = open-path travel time from the travel-time matrix, relative to the driver (negative = less). Official score = the challenge's score (0 = same order as the driver; lower is better).

| method | travel time vs driver | zone changes | official score | seconds/route* |
|---|---|---|---|---|
| driver | 0 | 20.3 | 0 | – |
| OR-Tools, 10 s | −10.38% | 35.8 | 0.0931 | 10.0 |
| **plain LNS** | **−13.80%** | 33.0 | 0.0798 | 2.0 |
| global SA | −12.30% | 34.5 | 0.0847 | 0.9 |
| zonal LNS, zone-level routing order | −11.29% | 29.2 | 0.0688 | 0.15 |
| zonal SA, zone-level routing order | −11.27% | 29.2 | 0.0697 | 0.7 |
| zonal LNS, reference-route order | −9.64% | 28.9 | 0.0432 | 0.14 |

With the zone-change penalty λ = 60 s: OR-Tools −9.37% / 0.0642; plain LNS −11.64% / 0.0580; zonal LNS with reference-route order −8.07% / **0.0349**; with a GNN order that also sees the reference route −8.18% / 0.0328.
*Clean single-process timings on 20 routes (`results/final_f4_timing.csv`).

Everything, with confidence intervals and the comparison with the published results (Cook, Held & Helsgaun 2022): **[docs/RESULTS.md](docs/RESULTS.md)**. Definitions and mathematics: **[docs/METHODS.md](docs/METHODS.md)**.

## Layout
```
src/lmr/        data.py, pathlns.py (LNS), fastlns.py (compiled LNS/SA), zoneplan.py (zone orders, windows), zone_gnn.py, gnn.py,
                methods.py (method specs), baselines.py (OR-Tools), literature.py (official score), evaluate.py, analysis.py
scripts/        run_specs.py (any method on any routes, in parallel), run_final.sh (all final runs), train_zone_gnn.py,
                solver_study.py, make_tables.py, make_figures.py
results/        CSVs behind every number (see results/README.md); archive_before_zone_fix/ is not used
tests/          38 tests (pytest)
```

## Run
```
pip install -e ".[gnn,baselines,dev]" numba        # numba is optional but makes LNS/SA 25-65x faster
export LMR_DATA_ROOT=/path/to/almrrc2021           # see DATA.md; the dataset is not in this repository
pytest -q
python scripts/run_specs.py --out results/demo.csv driver "lns:i=5000:kmax=2" "zlns:order=ztsp:w=3:ii=1000:wi=1500:kmax=2:l=60"
```
A method is a spec string, e.g. `zlns:order=ref:w=3:l=60` (see `src/lmr/methods.py`).
