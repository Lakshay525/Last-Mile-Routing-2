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

**Service times and time windows (evaluation only, never optimised).** For stop `i`: `s_i` = sum of the planned
service seconds of its packages; `[a_i, b_i]` = the tightest time window among its packages (latest start, earliest
end; undefined if no package has a window; about 63% of packages have none), in seconds after the route's departure.
Timeline: `t₀ = 0`; for each next stop `j` after `i`: `t ← t + T[i][j]`; if `t < a_j` wait until `a_j`; if `t > b_j` the
stop is **late** by `t − b_j`; then `t ← t + s_j`. We report route duration (`t` at the end: travel + service +
waiting), number of late stops and total lateness. Service time does not depend on the order, so it does not change
which route is best.

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
* **gnn:** the learned model of section 6.

## 6. Zone-order GNN (learned part of approach 1)

**Graph.** Per route, node 0 is the depot and nodes `1..Z` are zones. Zone IDs are *not* used as features because
they are too sparse (6,400 distinct IDs in 1,000 routes); the model sees geometry and travel times.

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

**Decoder (pointer network).** A GRU keeps state `g_t = GRU([x_{c_t}; x̄], g_{t−1})` where `c_t` is the current node and
`x̄` the mean embedding. Logit for candidate `j`: `u_j = 10·tanh( (W_q g_t)·(W_k x_j) / √d ) − β·M̃[c_t][j]`
(`β` learned), visited nodes masked, `p(j) = softmax(u)`. A route over zones is built node by node.

**Training: imitation only.** Loss = negative log-likelihood of the driver's zone order under teacher forcing,
averaged per step; Adam, learning rate 3·10⁻⁴, batch 16, early stopping on the validation Kendall τ between the
predicted and the driver's zone order. **No reinforcement learning is used** (REINFORCE was unstable on whole
routes, see `README.md`). Decoding is greedy.

Training routes are the train part of a 3,000-route sample; validation is its validation part; the evaluation
routes are the 207 test routes (the split is decided by an MD5 hash of the route ID, so no route is in two parts).

## 7. OR-Tools baseline (the baseline we aim to beat)

Google OR-Tools routing solver, modelled as one vehicle that starts at the depot and ends at an added dummy node
whose arcs cost 0 (this turns the open path into the solver's closed-route form). Costs `C` (the same penalised
matrix as the other methods) are multiplied by 10 and rounded to integers, because OR-Tools needs integer costs.
Search: first solution `PATH_CHEAPEST_ARC`, then `GUIDED_LOCAL_SEARCH` for `t` seconds of wall-clock time
(10 s unless stated). All other OR-Tools settings are the defaults; no tuning was done. Because the search is
time-limited, its result depends on how busy the machine is, so OR-Tools runs were done with nothing else running.

**LKH3 is not included.** It could not be installed with pip on this machine and the download was not authorised;
it is the stronger reference and remains future work.

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
* Financial cost is **not** reported: the cost parameters were not provided.
