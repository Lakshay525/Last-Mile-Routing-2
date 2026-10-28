# Methods: definitions and mathematics

Everything below is written from the code in `src/lmr/` and describes exactly what is run.
Section numbers are referred to from `RESULTS.md`.

## 1. Problem, notation and metrics

**Instance (one route).** Stops `V = {0, 1, …, n}` where `0` is the depot (the delivery station) and
`1..n` are delivery stops. `T[i][j]` is the travel time in seconds from `i` to `j`. It is **asymmetric**
(`T[i][j] ≠ T[j][i]` in general) and comes from the dataset. Every stop `i ≥ 1` has a zone label `z(i)` (Amazon's
`zone_id`); the depot has `z(0) = −1`. A route has `Z` distinct zones (about 20 on average, about 7 stops per zone).

**A route (solution)** is a permutation `π = (π₀, π₁, …, π_n)` of `V` with `π₀ = 0`. It is an **open path**: there
is no return to the depot (this follows the Amazon challenge).

**Travel time.**  `J_T(π) = Σ_{k=0}^{n−1} T[π_k][π_{k+1}]`.

**Zone changes.**  `S(π) = Σ_{k=1}^{n−1} 1[ z(π_k) ≠ z(π_{k+1}) ]`  (the depot edge is not counted).
A route that visits every zone in one block has `S = Z − 1`. We also report *blocks per zone* `(S + 1) / Z`
(1.0 means each zone is visited in a single block).

**Penalised objective.**  For a penalty `λ ≥ 0` (seconds per zone change)

    J_λ(π) = J_T(π) + λ · ( S(π) + 1 )

The `+1` is the depot→first-stop edge, which always changes zone; it is the same constant for every route and
changes nothing. Equivalently `J_λ(π) = Σ_k C[π_k][π_{k+1}]` with the **penalised matrix**
`C[i][j] = T[i][j] + λ · 1[z(i) ≠ z(j)]`. All optimisers below minimise `J_λ`; `λ = 0` is pure travel time.
**Primary target: minimise `J_T` (`λ = 0`).** `λ > 0` is used to study the time-versus-tidiness trade-off.

**Driver route.**  `π^d` is the order in which the real driver visited the stops (after re-indexing, `π^d = (0,1,…,n)`).

**Gap versus driver.**  `gap(π) = J_T(π) / J_T(π^d) − 1`. Negative = less travel time than the driver.
The dataset gives estimated travel times, not realised ones, so "less travel time" means "less according to the
travel-time matrix".

**Similarity to the driver.**  Kendall's τ between the positions of the stops in `π` and in `π^d` (1 = same
order, 0 = unrelated, −1 = reversed).  **Zone-order similarity**: the same τ computed on the sequence in which the
zones are *first entered* (needs `Z ≥ 2`).

## 2. Nearest neighbour (baseline)

From the current stop go to the unvisited stop with the smallest `T`. No search; used as a quick baseline and as
the starting route for LNS.

## 3. Plain LNS (approach 2)

Large neighbourhood search on the whole route, with the objective `J_λ` (matrix `C`).

* **State:** current route `p`, best route `p*`. Start: nearest neighbour under `C`.
* **Iteration.** Choose a destroy operator `d` and a repair operator `r` (probabilities proportional to learned
  weights), remove `k` stops, reinsert them, evaluate `J_λ`, accept or reject.
* **Number removed:** `k ~ Uniform{ max(2, ⌊m/40⌋), …, max(3, ⌊m/8⌋) }`, `m` = number of movable stops (about 148 → 3 to 18).
* **Destroy operators.**
  *random*: `k` random stops. *related (Shaw)*: pick a random stop `s`, remove the `k` stops with the smallest
  `T[s][x] + T[x][s]`. *worst*: remove the `k` stops with the largest detour
  `C[a][x] + C[x][b] − C[a][b]` (with `a`, `b` the neighbours of `x` in the route). *zone*: remove up to `k` stops of one zone.
* **Repair operators** (applied until all removed stops are back). The cost of inserting `x` between `a` and `b` is
  `Δ(x; a, b) = C[a][x] + C[x][b] − C[a][b]` (or `C[a][x]` at the end of an open path).
  *greedy*: repeatedly insert the (stop, position) pair with the smallest `Δ`.
  *regret-2*: repeatedly insert the stop with the largest `(second-best Δ) − (best Δ)` at its best position.
* **Acceptance (simulated-annealing rule):** accept the new route `q` if `J(q) < J(p)`, otherwise with probability
  `exp(−(J(q) − J(p)) / τ)`. Temperature `τ₀ = 0.02 · J(p₀) / n`, geometric cooling to `0.01` over the run.
* **Adaptive weights (the only "agent"-like element).** After each iteration both operator weights are updated
  `w ← 0.9 w + 0.1 (ρ + 0.05)` with reward `ρ = 5` (new best), `2` (improves the current route), `0.5` (accepted),
  `0` otherwise. In an earlier small run the adaptive weights gave no measurable benefit.
* **Budget:** number of iterations `i`. Run time grows roughly linearly with `i` (about 0.8 s for `i = 1500`
  on a 150-stop route on this machine).

## 4. Zone-specific LNS with overlapping zones (approach 1)

Three stages. The same LNS engine (section 3) is used inside each stage.

1. **Zone order** `ω = (ω₁, …, ω_Z)` (section 5).
2. **Solve each zone.** For `k = 1..Z`: the sub-problem is the path that starts at the stop where the previous zone
   ended and visits all stops of zone `ω_k`; it is solved by LNS (`ii` iterations; nearest-neighbour start). The
   concatenation gives a route in which every zone is one contiguous block `B_k` (a range of positions).
3. **Overlapping windows.** For a window width `w` (in zones) and stride 1, window `k` is the union of blocks
   `B_k … B_{k+w−1}` (a range of route positions). The stops in the window are re-optimised by LNS (`wi`
   iterations) with the stop **before** the window and the stop **after** it held fixed, under the same `J_λ`.
   Consecutive windows share `w − 1` zones, which is the **overlap**: a stop near a zone border can move into the
   neighbouring zone's part of the route, and the sub-problem re-solved after zone `k+1` has been optimised sees the
   new entry and exit stops. `w = 1` means no overlap (each zone re-solved once more with its exit fixed);
   `w ≥ 2` gives overlap. `s` sweeps repeat the pass over all windows.
   With `λ = 0` a window may mix zones freely; with `λ > 0` a stop only changes zone if the travel-time saving is
   larger than the penalty, so `λ` controls how far routes leave the zone structure ("not overwhelming but a cost").
4. **Result:** the final route; it is always a valid permutation (checked in the tests and in every experiment).

Because windows only reorder stops *within* a window, the zone order fixes the coarse structure of the route; the
quality of the zone order therefore limits what the method can reach.

## 5. Zone orders

* **greedy:** from the stops of the zone just left (initially the depot) go to the unvisited zone with the smallest
  mean travel time `mean_{i∈A, j∈B} T[i][j]`.
* **ztsp:** build the zone-level matrix `M[A][B] = mean_{i∈A, j∈B} T[i][j]` (depot as node 0) and solve the resulting
  small routing problem with LNS.
* **driver (oracle):** the order in which the driver first entered each zone. **Not a deployable method**; it is an
  upper bound on what any zone-order model could contribute.
* **random:** a random permutation of the zones (sanity check; should be clearly worse).
* **ref (reference route; from Cook, Held & Helsgaun 2022, section 4.5):** among the training routes of the same depot (never the route
  itself; the training split only), take the one that maximises `w(Q) · |zones(Q) ∩ zones(R)|` with `w` = 2 / 1.5 / 1 for driver quality High /
  Medium / Low, and follow the order in which that route's driver first entered the zones the two routes share. Zones of R that the reference
  never visited are inserted one at a time (largest first) at the position that adds the least zone-to-zone travel time.
* **gnn:** the learned model of section 6, optionally given the reference route's order as input (section 6).

## 6. Zone-order GNN (learned part of approach 1)

**Graph.** Per route, node 0 is the depot and nodes `1..Z` are zones. Zone IDs themselves are not features (too sparse: 6,400 distinct IDs in
1,000 routes); the model sees geometry and travel times, and optionally the order of the best reference route (below).

**Missing zone IDs.** 0.7% of stops (in 42% of routes) have no zone ID in the data. Each takes the zone of the nearest stop of the same route
(Euclidean on latitude and longitude); the paper does the same but searches the stops of the same depot over all routes.

* Node features (13): normalised centroid coordinates (2); mean/min/max of the zone's outgoing and incoming
  normalised zone-to-zone travel times (6); travel time from and to the depot (2); distance to the route centroid (1);
  share of the route's stops in the zone (1); depot indicator (1).
* Edge features (5): `M̃[A][B]`, `M̃[B][A]` (normalised by the mean off-diagonal value), whether the two zone IDs
  share the first-level prefix (e.g. `A-1`), whether they share the second-level prefix (e.g. `A-1.2`),
  normalised centroid distance. (Zone IDs look hierarchical; in the data, consecutive zones in the driver's order
  share the first-level prefix 90.6% of the time against 66.0% for a random order.)

**Encoder:** three edge-aware GATv2 layers. For node `i`, head `h`, neighbour `j`:
`s_ij^h = a_hᵀ LeakyReLU( W_l x_i + W_r x_j + W_e e_ij )`, `α_ij^h = softmax_j(s_ij^h)`,
`m_i = Σ_j α_ij W_r x_j`, `x_i' = LayerNorm( x_i + Dropout( ELU( W_o m_i ) ) )` (4 heads, width 64).

**Reference-route features (variant "GNN + reference").** Node features gain three entries: whether the best reference route visited the zone,
its relative rank among the shared zones (0 to 1), and the share of the route's zones the reference covers; edge features gain one entry:
+1 / −1 if the reference visited the row zone before / after the column zone, 0 otherwise. The model can then follow the reference where
it has evidence and fall back on geometry for the zones the reference never visited. Training and evaluation use the leave-one-out
reference of section 5.

**Decoder (pointer network).** A GRU keeps state `g_t = GRU([x_{c_t}; x̄], g_{t−1})` where `c_t` is the current node and
`x̄` the mean embedding. Logit for candidate `j`: `u_j = 10·tanh( (W_q g_t)·(W_k x_j) / √d ) − β·M̃[c_t][j]`
(`β` learned), visited nodes masked, `p(j) = softmax(u)`. A route over zones is built node by node.

**Training: imitation only.** Loss = negative log-likelihood of the driver's zone order under teacher forcing,
averaged per step; Adam, learning rate 3·10⁻⁴, batch 16, early stopping on the validation Kendall τ between the
predicted and the driver's zone order. **No reinforcement learning is used** (REINFORCE was unstable on whole
routes, see `README.md`). Decoding is greedy.

Training routes are the train part of all 6,112 routes (3,691 routes), validation is its validation part (1,213 routes) and early stopping uses the
validation teacher-forced next-zone accuracy; the evaluation routes are the 207 test routes (the split is decided by an MD5 hash of the route ID,
so no route is in two parts). Reference routes are drawn from the training part only.

## 7. OR-Tools baseline (the baseline we aim to beat)

Google OR-Tools routing solver, modelled as one vehicle that starts at the depot and ends at an added dummy node
whose arcs cost 0 (this turns the open path into the solver's closed-route form). Costs `C` (the same penalised
matrix as the other methods) are multiplied by 10 and rounded to integers, because OR-Tools needs integer costs.
Search: first solution `PATH_CHEAPEST_ARC`, then `GUIDED_LOCAL_SEARCH` for `t` seconds of wall-clock time
(10 s unless stated). All other OR-Tools settings are the defaults; no tuning was done. Because the search is
time-limited, its result depends on how busy the machine is, so OR-Tools runs were done with nothing else running.

## 8. Evaluation protocol

* Routes: the 207 test routes (hash bucket 80–99) of a 1,000-route sample of the 6,112 training routes.
* All methods are run on the same routes with the same seed (0); a method is only reported if it returned a valid
  permutation on every route.
* **Paired difference** of method `a` minus method `b`: for each route `(J_T(a) − J_T(b)) / J_T(driver) × 100`
  (percentage points of the driver's travel time). Report the mean, the 95% interval `±1.96·SE` (normal
  approximation, `SE = sd/√n`), and the share of routes where `a` is strictly better. If the interval excludes 0, the
  difference is treated as real.
* Run time: wall-clock seconds per route inside a worker process. Runs were executed with several worker processes in
  parallel, so absolute seconds are approximate; clean single-process timings are reported separately.

---

# Additions (second iteration)

## 9. Compiled LNS engine and tuned neighbourhood size

The LNS of section 3 is re-implemented in compiled code (numba; `src/lmr/fastlns.py`) with the same destroy and repair
operators, acceptance rule and adaptive weights. It runs 25–65× faster at the same number of iterations (about 0.05 s
for 1,500 iterations on a 150-stop route against about 1.2 s in Python). The Python engine remains as a fallback
(`LMR_ENGINE=py`); the random streams differ, so results agree in distribution, not run for run.

**Tuning (validation routes only, never the test routes).** The parameters were varied one at a time on 60 validation
routes. The number of removed stops is `k ~ Uniform{ max(2, ⌊m/kmin⌋), …, max(3, ⌊m/kmax⌋) }`. The original setting was
`kmin = 40, kmax = 8` (k up to about 18 of 150 stops). The only parameter that mattered was `kmax`: **`kmax = 2`** (up to
about half of the stops removed in one iteration) was better by about 1.4 points of travel time at equal run time. The
starting temperature `t0` (0.005–0.05), the end temperature, and `kmin` made no measurable difference. The chosen
setting is `i = 5000` iterations, `kmax = 2` (everything else unchanged). A 2-opt / Or-opt local-search polish after
LNS (`fastlns.local_search_nb`) was also tested; it adds +0.03 to +0.1 points and is optional (`pol=1`).

## 10. Simulated annealing (third approach)

Compiled SA on a path with the objective `Σ C` (same matrix C, same fixed endpoints as LNS). One iteration proposes
one move: **2-opt** segment reversal (probability 0.4), **Or-opt** relocation of a segment of 1–3 stops (0.4) or a
**swap** of two stops (0.2). With `Δ` the change in cost, the move is accepted if `Δ < 0` or with probability
`exp(−Δ / τ)`. The temperature falls geometrically from `τ₀ = t0 · J(p₀) / n` to `tf` over the run
(`t0 = 0.05` by default, `tf = 0.001`; `t0` was tuned on validation routes, see `RESULTS.md`). For an asymmetric
matrix a reversal changes the cost of the reversed segment, so the 2-opt delta is computed from both directions of
every edge in the segment.

**Zonal SA.** The zone-specific pipeline of section 4 with SA as the optimiser inside each zone and each window
(spec `zsa:`); `ii` and `wi` are then numbers of SA iterations (SA iterations are about 100× cheaper than LNS ones).

## 11. Beam search for the zone order, and a soft order constraint

* **Beam search.** Greedy decoding of the pointer network takes the most likely next zone at every step, so one early
  mistake changes everything after it. Beam search keeps the `B` most probable partial sequences at every step and
  returns the sequence with the highest total log-probability (`B = 1` is greedy decoding).
* **Order penalty ρ.** For a zone order ω with rank `rank(z)`, add `ρ · 1[ rank(z(i)) > rank(z(j)) ]` to `C[i][j]`: every
  edge that goes back to an earlier zone costs ρ seconds. With the zone-specific pipeline it keeps the order as a
  soft constraint (ρ = 0: no constraint; ρ large: strict order).

## 12. Closed tours and the official challenge score (comparison with the literature)

* **Closed tour.** The literature compares *tours* that return to the depot. `cl=1` solves the closed problem by adding
  a copy of the depot as a fixed last node (`T[i, depot copy] = T[i, 0]`, `T[depot copy, ·] = T[0, ·]`).
* **High+Delivered test set.** The 1,107 routes rated "High" with no `DELIVERY_ATTEMPTED` package (this definition
  reproduces the paper's count of 1,107 exactly; requiring every package `DELIVERED` would give 1,102).
* **Official score**, re-implemented from the public scoring code of the challenge (MIT-CAVE/rc-cli,
  `scoring/score.py`; the code was read, not copied). For the driver sequence `A` and a proposed sequence `B` on the same
  `n` stops (depot at both ends of each list):
  `SD(A,B) = 2 / (n(n−1)) · Σ_k ( |pos_A(B_{k+1}) − pos_A(B_k)| − 1 )` over the interior stops;
  the travel-time matrix is z-scored over all its entries and shifted so the minimum is 0 (equivalently `T / std(T)`);
  `ERP` is the edit distance between the two full lists with substitution cost `nm[a][b]` (row from `A`, column from `B`),
  gap cost `g = 1000`, ties broken in favour of the substitution/match; `ERP_per_edit = ERP / (number of edit
  operations)` (0 if none); **score = SD × ERP_per_edit**. Lower is better; the driver's own order scores 0.
  `docs/RESULTS.md` compares this implementation with the paper's published scores.

## 13. OR-Tools solvers compared (CP-SAT, MIP, routing)

* **Routing solver** (section 7): guided local search over OR-Tools' vehicle-routing model.
* **CP-SAT**: a Boolean variable per arc, `AddCircuit` (each node has one predecessor and one successor and the arcs form
  a single cycle), objective `Σ cost · arc` with costs ×10 rounded to integers. The open path is closed through a dummy
  node (stop → dummy at cost 0, dummy → depot forced). Optionally warm-started from an LNS route (`AddHint`). CP-SAT also
  reports a proven lower bound.
* **MIP**: the textbook compact formulation with Miller–Tucker–Zemlin order variables `u_i` and constraints
  `u_i − u_j + n x_ij ≤ n − 1` to forbid subtours, solved by SCIP through OR-Tools' linear-solver wrapper.
* `scripts/solver_study.py` runs them on small instances (first `k` stops of validation routes) and full routes.

## 14. Clusters inside the zone ID (extra penalty above the zone level)

A zone ID has the form `G-x.yD` (for example `A-2.1E`): a capital letter `G`, an integer `x`, an integer `y` and a capital letter `D`
(8,962 distinct IDs in the 6,112 training routes, two of which, `RTED3` and `LOA1`, do not fit the pattern and are treated as their own cluster).
Following Cook, Held & Helsgaun (2022, section 4.6), stops are also grouped into **clusters that agree on a subset of the four components**.
A chain of three nested clusterings `S > T > U` (`|S| = 3`, `|T| = 2`, `|U| = 1` components) is chosen on the **training routes only** as the one
the drivers cross least. For a route and a clustering `K`, the *excess crossings* of the driver are
`(number of consecutive stops in different clusters) − (number of clusters − 1)`, and the chain minimises the sum over its three levels of the
mean excess crossings per route. Result: **`S = {G, x, D}`, `T = {G, x}`, `U = {G}`** with 1.98, 0.73 and 0.04 excess crossings per route
(for comparison, 1.09 at the full-zone level), i.e. drivers almost never leave a first-letter cluster. This is the same chain as the example in the paper.

The extra penalty adds `hl` seconds for each of the three levels left: `C[i][j] ← C[i][j] + hl · (1[S(i)≠S(j)] + 1[T(i)≠T(j)] + 1[U(i)≠U(j)])`,
on top of `λ · 1[zone(i) ≠ zone(j)]` (spec key `hl`; code in `src/lmr/hierarchy.py`).
