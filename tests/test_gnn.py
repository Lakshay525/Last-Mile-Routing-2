import numpy as np, pytest
pytest.importorskip("torch")
from lmr.data import Route
from lmr.gnn import Policy, featurize

def test_policy_returns_valid_tour():
    n = 10; rng = np.random.default_rng(0)
    T = rng.uniform(10, 100, (n, n)); np.fill_diagonal(T, 0)
    r = Route("r", list(range(n)), rng.random((n, 2)), np.array([-1] + [i // 3 for i in range(n - 1)]),
              [], T, np.arange(n), "High", "X")
    x, e, Tn = featurize(r)
    seq, _ = Policy()(x, e, Tn, "greedy")
    assert sorted(seq[0].tolist()) == list(range(n)) and seq[0, 0].item() == 0