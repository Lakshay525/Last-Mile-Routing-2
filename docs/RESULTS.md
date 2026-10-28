# Results: three approaches against OR-Tools, and a comparison with the literature

Branch `zone-lns`. Definitions and mathematics: [METHODS.md](METHODS.md). **Every table below is generated from the CSV files in
`results/` by `scripts/make_tables.py`; no number was typed by hand.** Commands that reproduce each file: `scripts/run_final.sh`.

## 0. Read this first

**The three approaches**
1. **Approach 1, zonal LNS:** choose the order of the zones, solve each zone, then re-optimise overlapping windows of 3 zones with LNS (METHODS §4–5).
2. **Approach 2, plain (global) LNS:** one LNS over the whole route (METHODS §3).
3. **Approach 3, zonal SA** (the professor's suggestion): approach 1 with simulated annealing as the optimiser (METHODS §10). Global SA is reported as well.
**Baseline to beat:** Google OR-Tools (guided local search). **Reference:** the published results of Cook, Held & Helsgaun (2022), the team that won the challenge (section 4).

**Setup.** 207 held-out test routes (about 149 stops and 20 zones each), one random seed, all methods on the same routes. "Travel time vs driver" is
the open-path travel time from the travel-time matrix relative to the driver's route on the same route (−10% = 10% less). Intervals are mean ± 1.96·SE over routes (route
variation only; one seed). Parameters of every method were chosen on **validation** routes (section 6), never on the test routes.

**Corrections made on the way (so older numbers must not be used).** (a) 42% of routes have at least one stop without a zone ID (0.7% of stops). They had been treated
as the depot's zone; they now take the zone of the nearest stop, as in the LKH-AMZ paper. Every zone-dependent result was rerun. (b) The LNS was re-implemented in compiled code
(25–65× faster at equal quality) and its neighbourhood size was tuned. (c) The first OR-Tools run was repeated: a time-limited search varies from run to run, and both runs are shown.

## 1. Main findings

1. **Plain LNS finds the shortest routes.** −13.80% travel time against the driver, and 3.42 ± 0.27 points better than OR-Tools at 10 s (better on 98% of routes); single-process run time 2.0 s against 10 s (T1, T5, T7).
2. **Global SA is the weaker optimiser but still beats OR-Tools:** 1.50 ± 0.19 points worse than LNS, 1.92 ± 0.24 better than OR-Tools (T5).
3. **Zonal methods (approaches 1 and 3) are far faster and still beat OR-Tools on travel time when the zone order is chosen by zone-level routing:** −11.29% (0.91 ± 0.37 points better than OR-Tools 10 s) in 0.15 s (zonal LNS) or 0.72 s (zonal SA), clean timings (T1, T5, T7). **SA and LNS give the same result inside the zonal pipeline** (difference 0.02 ± 0.08).
4. **Zonal methods cost travel time against global LNS (2.52 ± 0.32 points for zone-level routing order) but produce routes much closer to the drivers' when the zone order comes from data.** With the zone order of the best-matching training route, the challenge's official score falls from 0.058 (plain LNS) and 0.064 (OR-Tools) to 0.035 at the balanced setting λ = 60 (differences −0.023 ± 0.004 and −0.029 ± 0.004, better on 80% and 89% of routes), at a cost of 3.57 ± 0.39 points of travel time against plain LNS (T2, T5).
5. **The zone order is the key ingredient.** The driver's own zone order (an oracle, not deployable) gives similarity 0.94 and official score 0.009; the data-derived orders reach 0.60 and 0.033–0.035; orders from geometry alone (nearest zone, zone-level routing) give ≈ 0 similarity (T2, T6).
6. **A GNN adds nothing measurable over copying the best reference route.** With the reference route's order as input the GNN predicts the driver's zone order as well as the reference does (Kendall τ +0.624 vs +0.626 on average, median 0.872 vs 0.750), and the routes it yields score the same (difference −0.0021 ± 0.0023, T5, T6). Without that input (geometry and travel times only) the GNN is far weaker: validation next-zone accuracy 76% against 82%, and test τ +0.093 with greedy decoding and +0.071 with beam search (T6), against +0.57 and +0.62 with the reference features. Beam search helped the model with reference features (+0.572 → +0.624) but not the geometry-only one.
7. **The balance of time and driver-likeness is tunable, and the best frontier comes from combining a data-derived zone order with the soft order penalty ρ and a global refinement (T3, T4, `figures/score_vs_time.png`).** At about the same travel time (−12.7% for ρ = 30 s against −12.9% for plain LNS with λ = 30 s) its official score is 0.048 against 0.064; at ρ = 100 s it reaches −9.7% travel time with score 0.037, where plain LNS with the zone penalty needs λ = 120 s (−10.6%) to get down to 0.054.
8. **The savings are modest in the context of a whole route.** Travel time is 40% of a route (10,923 s of travel plus 16,297 s of service time per route; service time is the same for every method). Plain LNS saves 1,466 s of travel, which is 5.4% of the driver's 27,220 s.

## 2. Approach by approach

* **Approach 2 (plain LNS)** is the best optimiser here. The largest single gain came from allowing much larger neighbourhoods (removing up to half the stops per iteration; section 6). Adding a penalty λ per zone change gives a smooth trade-off (T3): λ = 30 s already removes 68% of the extra zone changes (33.0 → 24.3 against the driver's 20.3) at a cost of 0.9 points of travel time.
* **Approach 1 (zonal LNS)** reaches OR-Tools-level or better travel time in about 1/70 of the time, and is the only method here that produces driver-like routes, but only if its zone order comes from the training data. It is clearly worse than plain LNS on travel time.
* **Approach 3 (zonal SA)** is, in this pipeline, indistinguishable from approach 1. SA only matters globally, where it is weaker than LNS.
* **OR-Tools** was run with one fixed configuration (guided local search). A tuned configuration, other solvers or longer runs were not explored beyond section 5.

## 3. Soft zone order and zone penalties

With a data-derived zone order the penalty ρ on edges that go back to an earlier zone (T4) trades travel time against similarity to the driver: ρ = 30 s gives −12.73% travel time at similarity 0.51 and score 0.048; ρ = 300 s gives −6.82% at 0.62 and 0.032. (All with the global refinement stage.)

## 4. Comparison with the literature (1,107 High+Delivered routes)

The test set of the LKH-AMZ paper was reproduced exactly (1,107 routes: High quality and no DELIVERY_ATTEMPTED package). Three published figures are reproduced to the digit or within a trial's noise:
the driver's mean closed-tour time (12,250.0 s), the number of stops (163,177) and the official score of random tours (0.9127 vs 0.91545). This validates the data handling, the closed-tour definition and our re-implementation of the official score.
Our near-optimal closed tours are **0.9% above the proven-optimal mean travel time** of the paper (10,958 s and 10,945 s at 5,000 and 15,000 iterations, against 10,853.3 s), and our zone-clustered tours are 0.6–0.8% above their optimal clustered tours
(11,298–11,330 s against 11,235.1 s). With a reference-route zone order our pipeline scores 0.0347 against 0.0317–0.0332 for LKH-AMZ with learned precedence constraints; with the driver's zone order 0.0079 against 0.0051–0.0074.

## 5. What is inside OR-Tools (CP-SAT vs MIP vs routing solver vs our LNS)

* On **small** instances CP-SAT proves optimality quickly (all 15 instances up to 20 stops, 9 of 15 at 30 stops, 4 of 15 at 40 in 20 s with one worker). The textbook MIP formulation (Miller–Tucker–Zemlin, SCIP) is slower at 20–30 stops and finds the best solution less often from 25 stops, but proves optimality on more 40-stop instances than CP-SAT (7 vs 4). (T8)
* **LNS** found the best known solution on 40–67% of the instances with 15–40 stops (mean excess 0.4–0.7%): it is not exact even on 15 stops, which is consistent with its 0.6% average gap to CP-SAT on full-size routes.
* On **full-size** routes CP-SAT, warm-started with our LNS route, improves it by only 0.61% on average in 60 s (max 1.9%) and proves optimality on 2 of 20; its proven lower bound is far too weak at this size (median 24% below) to certify near-optimality. Together with the literature anchor (0.9% above optimal), LNS is within about 1% of optimal.
* CP-SAT and the routing solver were **not** compared at equal time on full-size routes; the routing solver is the baseline in all other tables.
* A possible follow-up from the professor's suggestion, not done here: solving the windows of the zonal method with CP-SAT.

## 6. How the settings were chosen (validation routes only)

* **LNS neighbourhood:** the number of removed stops is drawn up to `m/kmax`. With `kmax = 2` (up to half of the stops) 5,000 iterations reached −14.07% on 60 validation routes, while the original `kmax = 8` reached only −12.69% at 15,000 iterations, −12.81% at 50,000 and −12.84% at 150,000: the larger neighbourhood beat the default even when the default was given 10–30× more iterations. The starting temperature, end temperature and minimum removal size made no measurable difference.
* **SA temperature:** a hotter start was clearly better (t0 = 0.05: −10.50%, 0.2: −11.71%, 0.5: −12.51%, 1: −12.33%, 2: −12.43%); `t0 = 0.5` was used.
* **Local-search polish after LNS** added only about 0.03 points and is not used.
* **Zonal settings** (windows of 3 zones; 1,000 iterations per zone and 1,500 per window for LNS, 50,000 and 300,000 for SA) were chosen on validation routes **before** the missing-zone fix and were not re-tuned afterwards. The penalty values (λ, ρ) are a fixed grid, not tuned.
* Raw tuning outputs: `results/tuning/` (the files in `archive_before_zone_fix/` predate the fix and are not used).

## 7. Feasibility and limits

* Computationally feasible: 0.15 s (zonal LNS) to 2 s (plain LNS) per route on a laptop CPU; OR-Tools needs 10 s for a worse result.
* The reference-route zone order needs historical routes from the same depot (here 3,691 training routes); for a new depot no reference exists, and a GNN that sees only geometry and travel times predicts the zone order poorly (τ +0.07 to +0.09) and is not a substitute.
* All gains are measured on the travel-time matrix, which holds estimates (no parking, construction or real traffic), and on a route-level metric that omits service time (the same for all methods). Whether drivers would accept routes that differ from theirs is not tested; the official score is only a proxy for that.
* OR-Tools results varied between two runs by 0.4 points at 10 s and 1.5 points at 2 s.

## 8. Reproduce

```
pip install -e ".[gnn,baselines,dev]" numba
bash scripts/run_final.sh                       # test routes, OR-Tools, literature set, timings (several hours)
python scripts/train_zone_gnn.py --ref --out results/zone_gnn_ref.pt     # zone-order GNN with reference features
python scripts/solver_study.py --out results/exp40_solver_study            # CP-SAT / MIP / LNS
python scripts/make_tables.py > docs/RESULTS_tables.md                    # all tables below
python scripts/make_figures.py
```

---

# Tables (generated)

## Test routes: 207 routes (hash split), seed 0

### T1. Travel time first (lambda = 0)

| method | travel time vs driver % | zone changes | similarity to driver | official score (lower=better) |
|---|---|---|---|---|
| driver | 0.00 | 20.3 | 1.000 | 0.0000 |
| nearest neighbour | 2.17 | 39.5 | -0.012 | 0.1118 |
| OR-Tools 2 s | -8.69 | 36.6 | -0.028 | 0.0978 |
| OR-Tools 10 s | -10.38 | 35.8 | -0.043 | 0.0931 |
| **plain LNS (approach 2)** | -13.80 | 33.0 | -0.031 | 0.0798 |
| global SA | -12.30 | 34.5 | 0.002 | 0.0847 |
| zonal LNS, zone-level routing order | -11.29 | 29.2 | -0.021 | 0.0688 |
| zonal LNS, reference-route order | -9.64 | 28.9 | 0.595 | 0.0432 |
| zonal LNS, GNN+reference order | -9.86 | 29.0 | 0.590 | 0.0416 |
| zonal SA (third approach), zone-level routing order | -11.27 | 29.2 | -0.019 | 0.0697 |
| zonal SA, reference-route order | -9.74 | 29.4 | 0.589 | 0.0448 |
| zonal LNS, driver's order (oracle, not deployable) | -9.49 | 29.3 | 0.920 | 0.0132 |

OR-Tools 10 s was run twice (time-limited, so results vary): run 1 -10.82% travel time vs driver; run 2 (table above) -10.38%.

### T2. Balanced objective (zone-change penalty lambda = 60 s)

| method | travel time vs driver % | zone changes | similarity to driver | official score (lower=better) |
|---|---|---|---|---|
| driver | 0.00 | 20.3 | 1.000 | 0.0000 |
| OR-Tools 10 s | -9.37 | 22.0 | 0.027 | 0.0642 |
| plain LNS | -11.64 | 21.3 | 0.011 | 0.0580 |
| global SA | -10.46 | 22.1 | 0.031 | 0.0628 |
| zonal LNS, zone-level routing order | -9.63 | 20.5 | -0.022 | 0.0553 |
| zonal LNS, greedy order | -7.35 | 20.4 | -0.019 | 0.0617 |
| zonal LNS, reference-route order | -8.07 | 20.5 | 0.605 | 0.0349 |
| zonal LNS, GNN+reference order | -8.18 | 20.5 | 0.600 | 0.0328 |
| zonal SA, reference-route order | -8.29 | 20.6 | 0.598 | 0.0357 |
| zonal SA, GNN+reference order | -8.44 | 20.7 | 0.592 | 0.0336 |
| zonal LNS, driver's order (oracle) | -7.77 | 20.5 | 0.939 | 0.0093 |

### T3. Zone-change penalty sweep, plain LNS

| method | travel time vs driver % | zone changes | similarity to driver | official score (lower=better) |
|---|---|---|---|---|
| driver | 0.00 | 20.3 | 1.000 | 0.0000 |
| lambda = 0 | -13.80 | 33.0 | -0.031 | 0.0798 |
| lambda = 15 | -13.46 | 27.6 | -0.004 | 0.0701 |
| lambda = 30 | -12.87 | 24.3 | -0.012 | 0.0638 |
| lambda = 60 | -11.64 | 21.3 | 0.011 | 0.0580 |
| lambda = 120 | -10.55 | 19.8 | -0.005 | 0.0541 |

### T4. Soft zone-order penalty rho (reference-route order, then global LNS refinement of 5,000 iterations)

| method | travel time vs driver % | zone changes | similarity to driver | official score (lower=better) |
|---|---|---|---|---|
| rho = 30 | -12.73 | 26.8 | 0.508 | 0.0478 |
| rho = 100 | -9.69 | 21.5 | 0.597 | 0.0367 |
| rho = 300 | -6.82 | 19.5 | 0.624 | 0.0320 |
| no refinement (windows only), lambda=0 | -9.64 | 28.9 | 0.595 | 0.0432 |
| GNN+ref, rho = 100 | -9.59 | 21.5 | 0.577 | 0.0366 |

### T5. Paired differences (percentage points of the driver's travel time; negative = first method has LESS travel time)

| comparison | mean difference (points) | 95% interval +- | first better on % of routes | n |
|---|---|---|---|---|
| plain LNS vs OR-Tools 10 s | -3.42 | 0.27 | 98 | 207 |
| global SA vs OR-Tools 10 s | -1.92 | 0.24 | 89 | 207 |
| global SA vs plain LNS | 1.50 | 0.19 | 11 | 207 |
| zonal LNS (ztsp) vs OR-Tools 10 s | -0.91 | 0.37 | 64 | 207 |
| zonal SA vs zonal LNS (same order) | 0.02 | 0.08 | 47 | 207 |
| zonal LNS (ztsp) vs plain LNS | 2.52 | 0.32 | 8 | 207 |
| plain LNS vs OR-Tools, lambda=60 | -2.28 | 0.26 | 90 | 207 |
| zonal LNS (ref) vs plain LNS, lambda=60 | 3.57 | 0.39 | 6 | 207 |
| zonal LNS (ref) vs OR-Tools, lambda=60 | 1.30 | 0.44 | 37 | 207 |
| zonal LNS: GNN+ref order vs ref order, lambda=60 | -0.11 | 0.26 | 34 | 207 |

Official score, paired (negative = first method scores better):

| comparison | mean difference | 95% interval +- | first better on % of routes | n |
|---|---|---|---|---|
| zonal LNS (ref) vs plain LNS, lambda=60 | -0.0231 | 0.0041 | 80 | 207 |
| zonal LNS (ref) vs OR-Tools, lambda=60 | -0.0293 | 0.0041 | 89 | 207 |
| GNN+ref vs ref order, lambda=60 | -0.0021 | 0.0023 | 46 | 207 |
| ref order vs zone-level routing order | -0.0204 | 0.0041 | 78 | 207 |

### T6. Zone-order prediction (Kendall tau between predicted and the driver's zone order, 207 test routes)

| order | mean | median | sd |
|---|---|---|---|
| GNN WITH reference-route features: gnn (greedy decoding) | 0.572 | 0.818 | 0.531 |
| GNN WITH reference-route features: gnn (beam 20) | 0.624 | 0.872 | 0.503 |
| greedy (nearest zone next) | -0.030 | -0.077 | 0.499 |
| ztsp (zone-level routing) | -0.037 | -0.105 | 0.608 |
| reference route (no learning) | 0.626 | 0.750 | 0.398 |
| random | -0.002 | 0.005 | 0.161 |
| GNN geometry only: gnn (greedy decoding) | 0.093 | 0.156 | 0.676 |
| GNN geometry only: gnn (beam 20) | 0.071 | 0.117 | 0.718 |

## Comparison with the literature: 1107 High+Delivered routes (paper: 1,107)

| method | mean closed-tour time (s) | official score | zone changes |
|---|---|---|---|
| closed LNS, no zone penalty (near-optimal ATSP tours) | 10958.0 | 0.07375 | 32.4 |
| closed LNS, 15,000 iterations | 10944.7 | 0.07366 | 32.3 |
| closed LNS, lambda=3000 (zone-clustered) | 11329.6 | 0.05031 | 19.2 |
| closed LNS, lambda=30000, 15,000 iterations (zone-clustered) | 11298.0 | 0.04939 | 19.2 |
| plain LNS lambda=60 (open path) | 11294.4 | 0.05175 | 20.8 |
| zonal LNS, reference-route order, lambda=60 | 11581.0 | 0.03470 | 20.1 |
| zonal LNS, driver's order (oracle), lambda=60 | 11591.4 | 0.00788 | 20.0 |
| zonal LNS, zone-level routing order, lambda=60 | 11485.8 | 0.04866 | 20.1 |
| global SA lambda=60 | 11415.4 | 0.05734 | 21.4 |
| driver | 12250.0 | 0.00000 | 20.0 |
| nearest neighbour | 12662.8 | 0.09969 | 38.4 |

Random tours, official score, mean of 10 trials: 0.91270 (paper 0.91545). Driver stops: 163177 (paper 163,177).
Paper (Cook et al.): optimal ATSP tours 10,853.3 s / 0.07030; optimal zone-clustered tours 11,235.1 s / 0.04866; driver zone order with LKH-AMZ 0.00510-0.00738; reference-route precedence constraints 0.03171-0.03322.

### T7. Clean timings (20 test routes, one process, nothing else running)

| method | mean_secs | median_secs | gap | score | routes |
|---|---|---|---|---|---|
| lns:i=5000:kmax=2 | 1.98 | 1.93 | -13.22 | 0.0766 | 20 |
| ortools:t=10 | 10.10 | 10.00 | -9.08 | 0.0915 | 20 |
| ortools:t=2 | 2.00 | 2.00 | -8.30 | 0.0936 | 20 |
| sa:i=5000000:t0=0.5 | 0.86 | 0.86 | -11.01 | 0.0852 | 20 |
| zlns:order=gnn:beam=20:w=3:ii=1000:wi=1500:kmax=2 | 0.35 | 0.33 | -9.28 | 0.0320 | 20 |
| zlns:order=ref:w=3:ii=1000:wi=1500:kmax=2 | 0.14 | 0.14 | -9.11 | 0.0388 | 20 |
| zlns:order=ztsp:w=3:ii=1000:wi=1500:kmax=2 | 0.15 | 0.15 | -9.95 | 0.0676 | 20 |
| zsa:order=ztsp:w=3:ii=50000:wi=300000:t0=0.5 | 0.72 | 0.71 | -9.94 | 0.0661 | 20 |

### T8. OR-Tools solvers on small instances (first k stops of validation routes, 20 s limit, 1 worker each)

| k | routes | CP-SAT proved optimal | MIP proved optimal | CP-SAT mean s | MIP mean s | LNS found best known % | CP-SAT found best known % | MIP found best known % | LNS excess over best % |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 15 | 15 | 15 | 0.5 | 0.4 | 100.0 | 100.0 | 100.0 | 0.0 |
| 15 | 15 | 15 | 15 | 0.1 | 0.9 | 66.7 | 100.0 | 100.0 | 0.4 |
| 20 | 15 | 15 | 13 | 0.6 | 5.8 | 60.0 | 100.0 | 100.0 | 0.6 |
| 25 | 15 | 14 | 11 | 4.6 | 9.6 | 40.0 | 100.0 | 73.3 | 0.7 |
| 30 | 15 | 9 | 9 | 9.5 | 12.9 | 53.3 | 100.0 | 66.7 | 0.5 |
| 40 | 15 | 4 | 7 | 16.9 | 15.4 | 40.0 | 66.7 | 46.7 | 0.5 |

Full-size routes (20 validation routes, CP-SAT 60 s, 2 workers, warm start from LNS): CP-SAT improves the LNS route by 0.61% on average (median 0.45%, max 1.92%); proved optimal on 2 of 20; median proven optimality gap 23.9%.
