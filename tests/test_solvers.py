import numpy as np
from lmr.solvers import nearest_neighbour, simulated_annealing, lns, zone_wise
from lmr.evaluate import cost

def toy(n=12, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.uniform(10, 100, (n, n)); np.fill_diagonal(T, 0)
    zone = np.array([-1] + [i // 4 for i in range(n - 1)])
    return T, zone

def valid(p, n): return sorted(p) == list(range(n)) and p[0] == 0

def test_nn():
    T, _ = toy(); assert valid(nearest_neighbour(T), len(T))

def test_sa_not_worse_than_start():
    T, _ = toy(); p, c = simulated_annealing(T, iters=2000)
    assert valid(p, len(T)) and c <= cost(T, nearest_neighbour(T)) + 1e-9

def test_lns():
    T, z = toy(); p, c, _ = lns(T, z, iters=50); assert valid(p, len(T))

def test_zone_wise():
    T, z = toy(); assert valid(zone_wise(T, z, simulated_annealing, iters=500), len(T))