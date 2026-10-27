import numpy as np
import pytest
from lmr.data import Route
from lmr.pathlns import penalised_matrix, path_cost, lns_path, plain_lns, nearest_neighbour_path
from lmr.zoneplan import (zone_members, zone_matrix, order_greedy, order_driver, order_random, order_ztsp,
                          zone_window_lns)
from lmr.methods import parse, run_spec
from lmr.evaluate import route_metrics, zone_switches


def toy_route(n=41, zones=6, seed=0):
    """Stops on a plane, zones = k-means-like angular sectors, travel time = distance (+ asymmetry)."""
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 10
    zone = np.concatenate([[-1], np.floor(np.arctan2(xy[1:, 1] - 5, xy[1:, 0] - 5) % (2 * np.pi) / (2 * np.pi) * zones)]).astype(int)
    D = np.linalg.norm(xy[:, None] - xy[None], axis=-1) * 20
    T = D * (1 + 0.1 * rng.random((n, n))); np.fill_diagonal(T, 0)
    names = [f"Z-{k}.1A" for k in range(zones)]
    return Route("toy", list(range(n)), xy, zone, names, T, np.arange(n), "High", "TST")


def valid(p, n): return sorted(p) == list(range(n)) and p[0] == 0


def test_penalised_matrix_equals_cost_plus_lambda_times_switches():
    r = toy_route(); lam = 37.0; C = penalised_matrix(r.T, r.zone, lam)
    p = list(np.random.default_rng(1).permutation(len(r.T)))
    p.remove(0); p = [0] + p
    expected = sum(r.T[a, b] for a, b in zip(p[:-1], p[1:])) + lam * sum(r.zone[a] != r.zone[b] for a, b in zip(p[:-1], p[1:]))
    assert path_cost(C, p) == pytest.approx(expected)
    # excluding the depot edge (a constant lam for every route) matches zone_switches
    assert path_cost(C, p) - path_cost(r.T, p) == pytest.approx(lam * (zone_switches(r.zone, p) + 1))


def test_lns_path_is_valid_and_not_worse_than_start():
    r = toy_route(); C = penalised_matrix(r.T, r.zone, 0); init = nearest_neighbour_path(C)
    p, c, _ = lns_path(r.T, C, r.zone, init, iters=200)
    assert valid(p, len(r.T)) and c <= path_cost(C, init) + 1e-9


def test_lns_path_keeps_fixed_end():
    r = toy_route(); C = penalised_matrix(r.T, r.zone, 0); init = list(range(len(r.T)))
    p, _, _ = lns_path(r.T, C, r.zone, init, iters=200, fixed_end=True)
    assert p[0] == 0 and p[-1] == len(r.T) - 1 and sorted(p) == init


def test_higher_penalty_gives_fewer_zone_switches():
    r = toy_route(n=61, zones=8)
    s0 = zone_switches(r.zone, plain_lns(r.T, r.zone, 0, 600))
    s1 = zone_switches(r.zone, plain_lns(r.T, r.zone, 500, 600))
    assert s1 <= s0 and s1 <= len(set(r.zone[1:])) + 2


@pytest.mark.parametrize("order", ["greedy", "driver", "random", "ztsp"])
@pytest.mark.parametrize("w", [1, 2, 3])
def test_zone_window_lns_is_valid(order, w):
    r = toy_route()
    o = {"greedy": order_greedy(r.T, r.zone), "driver": order_driver(r), "random": order_random(r.zone),
         "ztsp": order_ztsp(r.T, r.zone, iters=50)}[order]
    assert sorted(o) == sorted(zone_members(r.zone))
    p = zone_window_lns(r.T, r.zone, o, lam=30, w=w, sweeps=1, init_iters=30, win_iters=60)
    assert valid(p, len(r.T))


def test_zone_window_with_huge_penalty_visits_each_zone_once():
    r = toy_route(n=61, zones=6)
    p = zone_window_lns(r.T, r.zone, order_greedy(r.T, r.zone), lam=1e5, w=2, init_iters=50, win_iters=100)
    assert zone_switches(r.zone, p) == len(set(r.zone[1:])) - 1


def test_zone_matrix_shape_and_depot():
    r = toy_route(); m = zone_members(r.zone); zs = sorted(m); M = zone_matrix(r.T, m, zs)
    assert M.shape == (len(zs) + 1, len(zs) + 1) and np.all(np.diag(M) == 0)
    assert M[0, 1] == pytest.approx(r.T[0, m[zs[0]]].mean())


def test_spec_parser_and_methods():
    assert parse("zlns:order=gnn:w=2:l=60") == ("zlns", {"order": "gnn", "w": "2", "l": "60"})
    r = toy_route()
    for spec in ["driver", "nn", "lns:i=100:l=20", "zlns:order=greedy:w=2:l=20:ii=30:wi=60"]:
        assert valid(run_spec(spec, r), len(r.T)), spec


def test_route_metrics_driver_has_zero_gap():
    r = toy_route(); m = route_metrics(r, list(range(len(r.T))))
    assert m["gap_vs_driver"] == pytest.approx(0) and m["sim_to_driver"] == pytest.approx(1)


def test_ortools_baseline_valid_if_installed():
    pytest.importorskip("ortools")
    r = toy_route(n=31)
    p = run_spec("ortools:t=1", r)
    assert valid(p, len(r.T))


def test_lateness_penalty_matches_hand_computed_timeline():
    from lmr.pathlns import lateness_penalty
    T = np.array([[0, 10, 20], [10, 0, 5], [20, 5, 0]], dtype=float)
    nan = float("nan")
    pkg = dict(service=np.array([0.0, 4.0, 4.0]), tw_start=np.array([nan, nan, nan]), tw_end=np.array([nan, 100.0, 12.0]))
    f = lateness_penalty(T, pkg, mu=2.0)
    # path 0,1,2: arrive 1 at t=10 (ok, end 100), service -> 14, arrive 2 at t=19 > 12 -> late by 7 -> 2*7
    assert f(np.array([0, 1, 2])) == pytest.approx(14.0)
    # path 0,2,1: arrive 2 at t=20 > 12 -> late by 8, service -> 24, arrive 1 at 29 (ok) -> 2*8
    assert f(np.array([0, 2, 1])) == pytest.approx(16.0)


def test_plain_lns_with_lateness_penalty_is_valid():
    r = toy_route(n=31, zones=5); n = len(r.T)
    r.pkg = dict(service=np.full(n, 30.0), tw_start=np.full(n, np.nan), tw_end=np.full(n, np.nan))
    r.pkg["tw_end"][5] = 100.0
    assert valid(run_spec("lns:i=100:mu=3", r), n)


def test_sa_engine_valid_and_not_worse_than_start():
    from lmr.fastlns import sa_fast
    r = toy_route(n=31, zones=5); C = penalised_matrix(r.T, r.zone, 20.0); init = nearest_neighbour_path(C)
    for fe in (False, True):
        s0 = init if not fe else [x for x in init if x != len(C) - 1] + [len(C) - 1]
        p, c, _ = sa_fast(C, s0, 5000, 1, fe)
        assert valid(p, len(C)) and (not fe or p[-1] == len(C) - 1)
        assert c == pytest.approx(path_cost(C, p)) and c <= path_cost(C, s0) + 1e-9


def test_sa_specs_are_valid():
    r = toy_route(n=31, zones=5)
    for spec in ["sa:i=3000:l=10", "zsa:order=greedy:w=2:l=10:ii=500:wi=1000", "lns:i=100:cl=1", "sa:i=3000:cl=1"]:
        assert valid(run_spec(spec, r), len(r.T)), spec


def test_fix_missing_zones_assigns_nearest_stops_zone():
    from lmr.data import fix_missing_zones
    xy = np.array([[0, 0], [0, 1], [0, 2], [5, 5], [0, 1.9]]); zone = np.array([-1, 0, 0, 1, -1])
    out = fix_missing_zones(zone, xy)
    assert out[0] == -1 and out[4] == 0 and (out[1:4] == zone[1:4]).all()      # depot untouched; stop 4 is nearest to stop 2 (zone 0)


def test_official_score_is_zero_for_the_driver_and_matches_a_hand_computation():
    from lmr.literature import official_score
    T = np.array([[0, 5, 9, 4], [6, 0, 3, 8], [7, 2, 0, 5], [3, 9, 4, 0]], dtype=float)
    assert official_score(T, [0, 1, 2, 3]) == 0.0
    # proposed 0,2,1,3: interior positions in the driver's order are [1,0,2] -> sum(|diff|-1) = 1, SD = 2/(3*2) * 1 = 1/3
    # edits: positions 1 and 2 differ (2 edits), cost = nm[1][2] + nm[2][1] with nm = T/std(T); ERP per edit = that / 2
    nm = T / T.std()
    expected = (1 / 3) * (nm[1, 2] + nm[2, 1]) / 2
    assert official_score(T, [0, 2, 1, 3]) == pytest.approx(expected)


def test_official_score_random_order_is_worse_than_a_small_change():
    from lmr.literature import official_score
    r = toy_route(n=41); rng = np.random.default_rng(3)
    small = list(range(len(r.T))); small[5], small[6] = small[6], small[5]
    rand = [0] + list(rng.permutation(np.arange(1, len(r.T))))
    assert 0 < official_score(r.T, small) < official_score(r.T, rand)


def test_reference_order_is_a_valid_zone_order():
    from lmr.zoneplan import build_reference_index, order_reference, zone_members
    train = [toy_route(seed=s) for s in range(1, 4)]
    for i, q in enumerate(train): q.rid = f"q{i}"
    idx = build_reference_index(train)
    r = toy_route(seed=9); r.rid = "target"
    o = order_reference(r, idx)
    assert sorted(o) == sorted(zone_members(r.zone))


def test_soft_order_penalty_keeps_routes_valid_and_reduces_backward_moves():
    r = toy_route(n=61, zones=6); order = order_driver(r)
    p0 = zone_window_lns(r.T, r.zone, order, 0, w=2, init_iters=50, win_iters=100, global_iters=500, rho=0)
    p1 = zone_window_lns(r.T, r.zone, order, 0, w=2, init_iters=50, win_iters=100, global_iters=500, rho=1e5)
    rank = {z: i for i, z in enumerate(order)}
    back = lambda p: sum(rank[r.zone[b]] < rank[r.zone[a]] for a, b in zip(p[1:-1], p[2:]))
    assert valid(p0, len(r.T)) and valid(p1, len(r.T)) and back(p1) <= back(p0) and back(p1) == 0


def test_closed_tour_includes_the_return_leg():
    from lmr.literature import closed_time
    r = toy_route(n=31)
    p = run_spec("lns:i=300:cl=1", r)
    # the closed-tour search starts from the nearest-neighbour tour and keeps its best, so it can never end worse than it
    assert valid(p, len(r.T)) and closed_time(r.T, p) <= closed_time(r.T, nearest_neighbour_path(r.T)) + 1e-6
