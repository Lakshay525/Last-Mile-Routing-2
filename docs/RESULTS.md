# Results: plain LNS vs zone-specific LNS (GNN zone order, overlapping zones) vs OR-Tools

Branch `zone-lns`. Method definitions and formulas: `METHODS.md` (section numbers below refer to it).
Every number here comes from a CSV in `results/` (file named in each table). Nothing is rounded in our favour.

**Setup.** 207 test routes (hash bucket 80–99 of a 1,000-route sample of the 6,112 training routes), ~149 stops and
~20 zones per route. All methods use the same routes and seed 0. "Gap" = travel time relative to the real driver's
route on the same route (−12% = 12% less travel time according to the travel-time matrix). Intervals are
mean ± 1.96·SE over routes, so they show route-to-route variation only; **there is one random seed, so run-to-run
variation of the stochastic methods is not included.**

**Not done (and why).** LKH3: not installed, download not authorised. Financial cost: parameters not provided.
Official challenge score: not implemented. Full dataset: 207 of 6,112 routes evaluated, evaluation folder unused.
Solver parameters were not tuned.

## 0. What the main findings are

1. **Plain LNS finds the shortest routes and beats OR-Tools.** −12.0% vs the driver at 1,500 iterations; OR-Tools (10 s) −10.8%. Paired difference −1.18 ± 0.27 points, LNS better on 76% of routes. Even 500 LNS iterations (0.4 s) beat OR-Tools at 10 s (−0.59 ± 0.27).
2. **Zone-specific LNS does *not* beat plain LNS or OR-Tools on travel time.** Best variant (zone-level routing order, window of 3 zones) −9.2%; it is 1.65 ± 0.49 points worse than OR-Tools. Overlap clearly helps within the zone-specific method (no overlap −2.6%, 2 zones −5.3%, 3 zones −6.4%, greedy order).
3. **A zone-crossing penalty on plain LNS gives a better time/tidiness trade-off than the zone-specific method.** Plain LNS with λ=60 s: −9.9% travel time with 22.7 zone changes (driver: 21.9); with λ=120 s: −8.5% with 20.6. Zone-specific LNS at similar tidiness reaches −8.1% (22.6 changes) and −7.3% (20.9).
4. **The zone-order GNN is weak.** Its zone order correlates with the driver's at τ = +0.063 (sd 0.65 over routes), against +0.014 for a random order; that difference is not clearly real (≈0.05 ± 0.09). Used inside zone-specific LNS it gives the same travel time as the greedy order and is 2.6 ± 0.4 points worse than the zone-level-routing order.
5. **The driver's zone order is what makes routes look like the driver's.** With the driver's own zone order (oracle, not deployable), similarity to the driver's stop order is 0.944 (vs ≈ 0 for every other method) and late stops stay at driver level. A good zone-order predictor would therefore matter, but ours is far from the oracle.
6. **Ignoring time windows has a cost.** Unconstrained LNS has 0.70 late stops per route (driver 0.18). A lateness penalty in the acceptance rule brings that to 0.17–0.19 for 0.4–0.5 points of travel time.
7. **Perspective:** travel time is 40% of the route duration (service time, 16,297 s per route, is the same for every method). LNS's 12% travel saving is **4.6% of the driver's route duration**.

## 1. Baselines and plain LNS (λ = 0, primary target: travel time)  — `exp10_no_gnn.csv`, `exp13_ortools.csv`

| Method | Gap vs driver | Zone changes | Similarity to driver | Late stops / route |
|---|---|---|---|---|
| driver | 0 | 21.85 | 1.000 | 0.18 |
| nearest neighbour | +2.17% | 40.87 | −0.012 | 0.82 |
| OR-Tools, 2 s | −10.17% | 37.30 | −0.032 | 0.75 |
| OR-Tools, 10 s | −10.82% | 36.88 | −0.036 | 0.67 |
| LNS, 500 iterations | −11.41% | 36.19 | −0.015 | 0.71 |
| LNS, 1,500 iterations | −12.01% | 35.84 | −0.014 | 0.70 |
| LNS, 5,000 iterations | −12.26% | 35.62 | −0.005 | 0.74 |
| LNS, 15,000 iterations | −12.40% | 35.52 | −0.009 | 0.74 |

Diminishing returns: each 3–4× increase in iterations buys 0.15–0.6 points (paired: 500→1,500 −0.59 ± 0.12; 1,500→5,000 −0.26 ± 0.08; 5,000→15,000 −0.14 ± 0.06).

**Run times** (clean run: 20 routes, one process, nothing else running; `exp14_timing.csv`). Times in the other CSVs were measured with several worker processes and some with other jobs running, so they are inflated; use this table.

| Method | Mean seconds per route | Gap on these 20 routes |
|---|---|---|
| LNS 500 | 0.42 | −10.66% |
| LNS 1,500 | 1.33 | −11.43% |
| LNS 5,000 | 4.26 | −11.79% |
| zone-specific, zone-level-routing order, w=3 | 0.74 | −8.79% |
| zone-specific, GNN order, w=3 | 0.66 | −7.37% |
| zone-specific, greedy order, w=2 | 0.70 | −5.47% |
| OR-Tools, 2 s | 2.00 | −8.94% |
| OR-Tools, 10 s | 10.0 median (mean 29.3) | −9.38% |

The OR-Tools mean is inflated by one run that took 395 s (route `0f6118f8…`); the same route rerun alone took 10.3 s with the same quality, so this was a machine hiccup, not a property of the method.

**Paired comparison with OR-Tools 10 s** (percentage points of the driver's travel time; negative = first method better; `figures/vs_ortools.png`):

| First method | Difference | Better on |
|---|---|---|
| LNS 500 | −0.59 ± 0.27 | 65% of routes |
| LNS 1,500 | −1.18 ± 0.27 | 76% |
| LNS 15,000 | −1.58 ± 0.27 | 81% |
| LNS 1,500, λ=60 vs OR-Tools λ=60 | −0.40 ± 0.31 | 61% |
| zone-specific (zone-level routing, w=3) | +1.65 ± 0.49 | 35% |
| zone-specific (zone-level routing, w=3), λ=60 vs OR-Tools λ=60 | +2.22 ± 0.44 | 23% |

OR-Tools was used with one fixed configuration (guided local search, default settings); a tuned OR-Tools or longer runs could narrow the gap. LKH3, the stronger reference, is missing.

## 2. Zone-specific LNS, λ = 0 — `exp10_no_gnn.csv`, `exp12_gnn_order.csv`

Zone order → solve each zone → re-optimise overlapping windows of `w` zones (METHODS §4). w = 1 means no overlap.

| Zone order | w | Gap vs driver | Zone changes | Similarity to driver | Zone-order similarity |
|---|---|---|---|---|---|
| greedy | 1 | −2.60% | 19.73 | −0.013 | −0.034 |
| greedy | 2 | −5.30% | 27.59 | −0.015 | −0.034 |
| greedy | 3 | −6.39% | 29.41 | −0.015 | −0.033 |
| zone-level routing | 2 | −8.16% | 27.59 | −0.017 | −0.031 |
| zone-level routing | 3 | −9.17% | 29.44 | −0.018 | −0.030 |
| GNN | 2 | −5.53% | 27.48 | 0.073 | 0.062 |
| GNN | 3 | −7.16% | 30.23 | 0.072 | 0.061 |
| driver's (oracle) | 2 | −5.47% | 26.89 | **0.944** | **0.985** |
| random | 2 | +26.01% | 23.38 | 0.022 | 0.014 |

Paired differences (points): overlap w=2 vs w=1 (greedy) −2.70 ± 0.24, better on every route; w=3 vs w=2 −1.10 ± 0.16; zone-level routing vs greedy order (w=2) −2.87 ± 0.32; **driver's order vs greedy order −0.17 ± 0.46 (no difference in travel time)**; GNN vs greedy −0.24 ± 0.41 (no difference); GNN vs zone-level routing +2.63 ± 0.39 (GNN worse); random vs greedy +31.3 ± 2.3 (sanity check passes: a bad zone order costs a lot).

Reading: zone order matters a lot for travel time when it is bad (random), and for similarity to the driver when it is right (oracle), but the driver's zone order is *not* better for travel time than a simple rule. Zone-level routing gives the best travel time of the zone-specific variants.

## 3. Zone-crossing penalty (balance of time and tidiness) — `exp11_penalty.csv`, `exp12_gnn_order.csv`

λ = penalty in seconds per zone change (METHODS §1). Mean zone changes per route; the driver makes 21.85. Figure: `figures/tradeoff.png`.

| λ (s) | Plain LNS: gap / zone changes | Zone-specific (zone-level routing, w=3) | Zone-specific (GNN, w=3) |
|---|---|---|---|
| 0 | −12.01% / 35.84 | −9.17% / 29.44 | −7.16% / 30.23 |
| 15 | −11.95% / 30.01 | −8.72% / 24.80 | – |
| 30 | −11.35% / 26.31 | −8.07% / 22.56 | – |
| 60 | −9.92% / 22.71 | −7.30% / 20.89 | −5.09% / 21.16 |
| 120 | −8.45% / 20.64 | −6.45% / 20.02 | – |

Compared at matching numbers of zone changes, plain LNS keeps more of the travel-time saving: about 22.6 changes — plain LNS (λ=60) −9.92% vs zone-specific (λ=30) −8.07%, i.e. 1.9 points more; about 20.6–20.9 changes — plain LNS (λ=120) −8.45% vs zone-specific (λ=60) −7.30%, i.e. 1.2 points more. (These are comparisons of mean values at slightly different tidiness, not paired tests.) Relative to the unpenalised plain LNS (35.8 changes, 14 more than the driver), a penalty of λ=15 removes 42% of the extra zone changes at a cost of 0.06 points of travel time, and λ=30 removes 68% at a cost of 0.66 points. The "right" λ depends on how much a zone change is worth, which is a decision for the project, not something the data decides.

## 4. Zone-order GNN — `zone_gnn.json`, `zone_gnn_train.log`

Imitation of the driver's zone order only (no reinforcement learning). 1,815 training routes, 598 validation routes, 207 test routes; 22 epochs run (early stopping on validation τ), best validation τ = +0.088.

| Zone order | Test τ vs driver: mean | median | sd |
|---|---|---|---|
| GNN | +0.063 | +0.145 | 0.650 |
| greedy (nearest zone next) | −0.034 | −0.095 | 0.485 |
| zone-level routing | −0.033 | −0.116 | 0.591 |
| random | +0.014 | +0.014 | 0.166 |

The GNN is the best of the four on the median, but the mean difference to random is small relative to the spread (standard error of the GNN mean ≈ 0.045). In the downstream test (section 2) it raises similarity to the driver by +0.088 ± 0.083 over the greedy order and does not reduce travel time. **Conclusion: with these features and ~1,800 training routes the model has learned only a little about drivers' zone order.** Possible reasons (not tested): too little data; features that cannot capture drivers' habits (street layout, access); noise in the labels.

## 5. Service times and time windows (evaluation only, plus one penalised variant) — `exp10_no_gnn.csv`, `exp15_timewindows.csv`

Per route (means over 207 routes): driver travel time 10,923 s, service time 16,297 s (60% of the 27,393 s duration), 9.4 stops with a time window out of 149. Lateness is simulated with the matrix travel times and planned service times (METHODS §1); the driver's real lateness is not known.

| Method | Gap vs driver | Late stops / route | Late seconds / route |
|---|---|---|---|
| driver | 0 | 0.18 | 301 |
| OR-Tools 10 s | −10.82% | 0.67 | 2,450 |
| LNS 1,500 | −12.01% | 0.70 | 2,883 |
| LNS, lateness penalty μ=1 | −11.62% | 0.19 | 695 |
| LNS, lateness penalty μ=5 | −11.50% | 0.17 | 473 |
| LNS λ=60 | −9.92% | 0.71 | 2,553 |
| LNS λ=60, μ=5 | −9.61% | 0.25 | 784 |
| zone-specific, driver's order (oracle), w=2 | −5.47% | 0.21 | – |

μ = penalty per second of lateness, applied when accepting a move (so it does not steer where stops are inserted). Cost of μ=5 versus plain LNS: +0.51 ± 0.17 points. Only 6% of stops have a window, so this is a small effect on travel time but it removes most of the extra lateness. The oracle zone-specific route has driver-like lateness without any penalty, which suggests the driver's zone order carries information about the windows.

## 6. Honest assessment

* **Is the primary target met?** Yes for plain LNS: it reduces travel time against the driver by 12.0% ± 0.7 (95% interval over routes; shorter than the driver on 207 of 207 routes) and beats OR-Tools, in about 1/8 of its time in the clean timing (1.3 s vs 10 s). The saving is 4.6% of total route duration, and rests on the travel-time matrix being accurate (it holds estimated, not measured, travel times).
* **Is the zone-specific approach worth it?** On this evidence, not on travel time. It is not better than plain LNS, even with overlap and a zone-order model, and it is worse than OR-Tools. Its only demonstrated advantages are speed (0.7 s vs 1.3 s at the tested budgets, a small difference) and, with the *oracle* zone order, driver-like routes. A zone-order model that closes the gap to the oracle would change this, but ours does not (τ +0.063 vs 0.985).
* **Is the GNN useful?** Not yet. Used at route level earlier (`exp02_gnn.csv` of the main branch, 62 routes) it built routes 8.8% longer than the driver's and 7.1 ± 1.6 points worse than nearest neighbour, and gave no change as a warm start for SA or LNS; used at zone level it is better than the rules on the median but weak.
* **Open risks for a paper:** single seed; 207 test routes; untuned parameters; no LKH3; no official challenge score; the 12% claim is on a modelled quantity.

## 7. Reproduce

```
pip install -e ".[gnn,baselines,dev]"
python scripts/train_zone_gnn.py --epochs 30 --out results/zone_gnn.pt
python scripts/run_specs.py --out results/exp10_no_gnn.csv driver nn lns:i=500 lns:i=1500 lns:i=5000 lns:i=15000 zlns:order=greedy:w=1 zlns:order=greedy:w=2 zlns:order=greedy:w=3 zlns:order=ztsp:w=2 zlns:order=driver:w=2 zlns:order=random:w=2
python scripts/run_specs.py --out results/exp12_gnn_order.csv --zone-model results/zone_gnn.pt zlns:order=gnn:w=2:l=0 zlns:order=gnn:w=3:l=0 ...   # and l=15,30,60,120
python scripts/run_specs.py --out results/exp13_ortools.csv ortools:t=10 ortools:t=2 ortools:t=10:l=60     # nothing else running
python scripts/run_specs.py --out results/exp15_timewindows.csv lns:i=1500:mu=1 lns:i=1500:mu=5 lns:i=1500:l=60:mu=5
python scripts/make_figures.py
```
