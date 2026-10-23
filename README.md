# Last-mile routing: SA vs LNS, zone-wise vs whole-route, and GNN warm starts

> **Branch `zone-lns` (new work):** plain LNS and zone-specific LNS (GNN-predicted zone order, overlapping zone windows),
> zone-crossing penalty, OR-Tools baseline, service-time / time-window metrics.
> Method definitions and mathematics: [docs/METHODS.md](docs/METHODS.md). Results with every number traced to a CSV:
> [docs/RESULTS.md](docs/RESULTS.md). Run anything with `python scripts/run_specs.py --out results/x.csv <method specs>`
> (spec syntax in `src/lmr/methods.py`). LKH3 and the financial-cost metric are not included yet.


Deakin University research project on the last-mile routing problem, using the Amazon Last Mile Routing
Research Challenge 2021 data (Los Angeles, Seattle, Boston, Austin and Chicago stations; training set of 6,112 routes).

## Research questions
1. Does large neighbourhood search (LNS) find shorter routes than simulated annealing (SA) at equal compute time?
2. Is it better to plan a whole route at once, or one delivery zone at a time?
3. Can a graph neural network (GNN) + pointer network help, either by building routes itself or by giving
   SA/LNS a better starting route?
4. (Open) Can the zone order be learned from drivers?

## What is measured
All methods build a visiting order for one route's stops, starting at the depot. Each is compared with the order
the real driver followed on the same route.

| Metric | Meaning |
|---|---|
| `gap_vs_driver` | Route cost relative to the driver's route (-0.10 = 10% less driving time than the driver) |
| `zone_sw` | Number of times consecutive stops belong to different zones |
| `sim_to_driver` | Kendall tau between the method's stop order and the driver's order (1 = identical, 0 = unrelated) |
| `zone_order_sim` | Kendall tau between the order in which zones are first entered, vs the driver |
| `secs` | Planning time per route |

**Cost** is the sum of the travel times between consecutive stops, taken from the challenge travel-time matrix
(asymmetric: A to B can differ from B to A). The route is an open path from the depot, with no return leg.
Service time at stops and time windows are **not** included (see Limitations).

## Methods
| Name | Description |
|---|---|
| `driver` | The real route (reference) |
| `nn` | Nearest neighbour: always go to the closest unvisited stop |
| `sa` | Simulated annealing: 2-opt, segment-move and swap changes, geometric cooling, 750,000 iterations |
| `lns` | Adaptive large neighbourhood search: four removal rules (random, related, worst-detour, whole-zone), two repair rules (cheapest insertion, regret-2), operator weights learned online, 1,500 iterations |
| `lns_static` | LNS without adaptive weights (ablation; not yet run at scale) |
| `zone_sa` / `zone_lns` | Each delivery zone is solved as its own sub-problem, then the zones are joined (zones ordered greedily by travel time) |
| `gnn` | GNN encoder (3 edge-aware GATv2 layers) + GRU pointer-network decoder |
| `gnn_sa` / `gnn_lns` | SA or LNS started from the GNN's route instead of nearest-neighbour |

## Data and split
- 1,000 routes sampled from the training set (seed 0), split by an MD5 hash of the route ID into
  60% train / 20% validation / 20% test. The split does not depend on how many routes are loaded,
  so no route is ever in both training and test.
- Results below use the **207 test routes**.
- The dataset is not included. See [DATA.md](DATA.md).

## Results

### Classical methods (`results/exp01_classical.csv`, 207 test routes, seed 0)
| method | gap vs driver | zone switches | similarity to driver | secs/route |
|---|---|---|---|---|
| driver | 0 | 21.9 | 1.00 | - |
| lns | **-12.0%** | 35.8 | -0.01 | 13.1 |
| sa | -10.0% | 37.4 | -0.01 | 14.2 |
| nn | +2.2% | 40.9 | -0.01 | 0.0004 |
| zone_lns | +0.8% | **19.7** | -0.01 | **0.29** |
| zone_sa | +0.5% | 19.7 | -0.01 | 4.2 |

Zone-order similarity to the driver, 207 test routes: `zone_lns` mean -0.034 (median -0.095),
`nn` mean -0.010 (median -0.067); standard deviation about 0.48.

### GNN (`results/gnn_v2.history.json`, validation set)
Training cost is route length in units of the mean inter-stop travel time. References on validation:
nearest-neighbour 31.19, driver 30.54.

| Phase | Epochs | Validation cost | Similarity to driver |
|---|---|---|---|
| Imitation (copy the driver) | 0-2 | 31.91 to 32.95 | about 0.07 |
| REINFORCE | 3-5 | 35.38 to 37.29 (diverged) | about 0.07 |

The saved model is the end-of-imitation model (cost 32.95).

### GNN as a warm start (`results/exp02_gnn.csv`, 62 test routes)
| method | gap vs driver | zone switches | similarity | secs/route |
|---|---|---|---|---|
| driver | 0 | 22.1 | 1.00 | - |
| lns | -12.4% | 36.9 | -0.01 | 11.6 |
| gnn_lns | -12.5% | 36.1 | -0.01 | 11.5 |
| sa | -10.4% | 38.2 | -0.02 | 12.1 |
| gnn_sa | -10.8% | 37.2 | -0.01 | 11.8 |
| nn | +1.7% | 41.4 | -0.03 | 0.0005 |
| gnn | +8.8% | 44.0 | -0.02 | 0.07 |

Paired differences (percentage points, mean ± 95% interval): gnn_sa − sa = −0.4 ± 0.6; gnn_lns − lns = −0.04 ± 0.5;
lns − sa = −2.0 ± 0.5; gnn − nn = +7.1 ± 1.6.

## Findings
1. At about equal time (13-14 s per route), **LNS finds routes about 2 percentage points shorter than SA**
   (-12.0% vs -10.0% relative to the driver).
2. Both beat the real drivers on driving time. Plain nearest-neighbour is only 2.2% worse than the driver,
   so the drivers are not optimising this travel-time matrix.
3. Drivers are **zone-contiguous** (about 22 zone switches vs 36-41 for the unconstrained solvers).
   Zone-wise planning is within 1% of the driver's cost, with about 20 zone switches, and `zone_lns` does it
   in 0.29 s (about 45x faster than LNS). The price is giving up the 10-12% savings that LNS and SA find by crossing zones.
4. **No method reproduces the driver's visiting order** (similarity about 0), including zone-wise planning.
   The zone-order similarity is also about 0, so the mismatch includes *which zone is visited next*; the
   greedy "nearest zone next" rule is not what drivers do.
5. The GNN, trained by imitation, is about as long as nearest-neighbour or longer and only weakly
   driver-like (similarity 0.07). REINFORCE training diverged and is unresolved.
6. The GNN on its own builds routes about 9% longer than the driver's, worse than nearest-neighbour.
   Used as a warm start it makes no measurable difference to SA or LNS (paired differences are within noise).
   LNS beats SA by about 2 points on this second sample as well.

## Reproduce
```powershell
pip install -e ".[gnn,dev]"
$env:LMR_DATA_ROOT = "D:/path/to/almrrc2021"
python scripts/run_experiment.py --n-routes 1000 --methods driver,nn,sa,lns,zone_sa,zone_lns --out results/exp01_classical.csv
python scripts/train_gnn.py --n-routes 1000 --epochs 8 --out results/gnn_v2.pt
python scripts/run_experiment.py --n-routes 300 --methods driver,nn,sa,lns,gnn,gnn_sa,gnn_lns --gnn results/gnn_v2.pt --out results/exp02_gnn.csv
pytest -q
```
Notebooks in `notebooks/` read the CSVs in `results/`: dataset overview (01), classical baselines (02),
GNN (03), summary (04).

## Repository layout
```
src/lmr/        data.py, solvers.py, gnn.py, evaluate.py, agents.py, config.py
scripts/        run_experiment.py, train_gnn.py
tests/          unit tests (pytest)
notebooks/      analysis notebooks
results/        experiment outputs (CSV, JSON, training histories)
```

## Limitations
- **Driving time only.** Service time at stops, time windows and vehicle capacity are not modelled.
  Routes that are shorter by this measure may break a delivery constraint.
- **Single seed** for sampling and for the solvers; intervals across seeds are not yet reported.
- **Timings are approximate.** Some runs shared the CPU with other jobs. Costs, zone switches and similarity are unaffected.
- **Similarity** is a rank correlation over the full stop order, which penalises any local reordering.
  The official challenge score (sequence deviation plus an edit-distance penalty) is not yet implemented.
- **SA and LNS parameters are untuned** (temperatures, removal sizes).
- **The GNN is unfinished:** small training set, REINFORCE unstable, saved model is the imitation-only version.
- **Adaptive-LNS ablation** (`lns` vs `lns_static`) has not been run at scale; an earlier 40-route run showed no benefit.

## Next steps
- Learn the zone order from drivers, then plan inside zones with SA/LNS.
- Add a zone-switch penalty to the SA/LNS objective and report the cost-vs-tidiness trade-off.
- Debug REINFORCE on small instances, or train the GNN on LNS solutions instead of driver routes.
- Add the official challenge score, service times and time windows.

## Experiment log
- 2026-10-22: Classical comparison at about equal time on 207 test routes; GNN imitation + REINFORCE trained (v2); GNN warm-start experiment running.