"""Zone-level GNN: predicts the ORDER in which a route visits its ~20 zones.

Each route becomes a small graph: node 0 = depot, nodes 1..Z = zones. Training is imitation of the driver's
zone order (the order in which the driver first entered each zone). No reinforcement learning is used here.
Zone IDs cannot be used directly as features (6,400 distinct IDs in 1,000 routes); instead the model sees
geometry, zone-to-zone travel times, zone size, and whether two zone IDs share a prefix (the IDs look hierarchical).
"""
import copy, random
import numpy as np
import torch
from scipy.stats import kendalltau
from .zoneplan import zone_members, zone_matrix, order_driver
from .gnn import Policy

FIN, DE = 13, 5


def _prefix(name, level):
    head, _, tail = name.partition(".")
    return head if level == 1 else head + "." + tail[:-1]


def zone_instance(r):
    """Tensors for one route: (x [1,Z+1,FIN], e [1,Z+1,Z+1,DE], Tn [1,Z+1,Z+1]), zone ids by node, driver target."""
    members = zone_members(r.zone); zs = sorted(members); Z = len(zs)
    M = zone_matrix(r.T, members, zs)
    eye = np.eye(Z + 1, dtype=bool); Tn = M / M[~eye].mean()
    off = np.where(eye, np.nan, Tn)
    xy = np.vstack([r.xy[0]] + [r.xy[members[z]].mean(0) for z in zs])
    xyn = (xy - xy.mean(0)) / (xy.std(0).mean() + 1e-9)
    size = np.array([0.0] + [len(members[z]) for z in zs]); share = size / size.sum()
    isd = np.zeros(Z + 1); isd[0] = 1
    cen = np.linalg.norm(xyn, axis=1)
    x = np.column_stack([xyn, np.nanmean(off, 1), np.nanmin(off, 1), np.nanmax(off, 1),
                         np.nanmean(off, 0), np.nanmin(off, 0), np.nanmax(off, 0),
                         Tn[0], Tn[:, 0], cen, share, isd])
    names = [""] + [r.zone_names[z] for z in zs]
    p1 = np.array([_prefix(n, 1) if n else "#" for n in names]); p2 = np.array([_prefix(n, 2) if n else "#" for n in names])
    dist = np.linalg.norm(xyn[:, None] - xyn[None], axis=-1)
    e = np.stack([Tn, Tn.T, (p1[:, None] == p1[None]) & (p1[:, None] != "#"),
                  (p2[:, None] == p2[None]) & (p2[:, None] != "#"), dist / (dist.mean() + 1e-9)], -1).astype(np.float32)
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)[None]
    idx = {z: i + 1 for i, z in enumerate(zs)}
    target = torch.tensor([[0] + [idx[z] for z in order_driver(r)]], dtype=torch.long) if len(r.actual) else None
    return t(x), torch.tensor(e)[None], t(Tn), zs, target


@torch.no_grad()
def predict_order(pol, r):
    """Greedy zone order for route r, as a list of the route's zone ids."""
    x, e, Tn, zs, _ = zone_instance(r)
    seq = pol(x, e, Tn, "greedy")[0][0].tolist()
    return [zs[i - 1] for i in seq[1:]]


def order_tau(order_a, order_b):
    pos = {z: i for i, z in enumerate(order_b)}
    return kendalltau(range(len(order_a)), [pos[z] for z in order_a])[0]


def evaluate_orders(pol, routes):
    """Mean Kendall tau between the model's zone order and the driver's."""
    pol.eval(); return float(np.mean([order_tau(predict_order(pol, r), order_driver(r)) for r in routes]))


def train_zone_gnn(train_routes, val_routes, epochs=40, lr=3e-4, bs=16, patience=6, seed=0, verbose=True):
    torch.manual_seed(seed); random.seed(seed)
    pol = Policy(fin=FIN, de=DE); opt = torch.optim.Adam(pol.parameters(), lr)
    data = [zone_instance(r) for r in train_routes]
    best, best_state, bad, hist = -9, None, 0, []
    for ep in range(epochs):
        random.shuffle(data); pol.train(); tot = 0.0
        for i in range(0, len(data), bs):
            loss = 0
            for x, e, Tn, _, tgt in data[i:i + bs]:
                _, lp = pol(x, e, Tn, target=tgt); loss = loss - lp.mean() / (tgt.shape[1] - 1)   # per-step NLL
            opt.zero_grad(); (loss / len(data[i:i + bs])).backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step(); tot += float(loss.detach())
        tau = evaluate_orders(pol, val_routes)
        hist.append(dict(epoch=ep, train_nll=tot / len(data), val_tau=tau))
        if verbose: print(f"epoch {ep:2d}  train NLL/step {tot / len(data):.3f}  val zone-order tau {tau:+.3f}", flush=True)
        if tau > best: best, best_state, bad = tau, copy.deepcopy(pol.state_dict()), 0
        else:
            bad += 1
            if bad >= patience: break
    pol.load_state_dict(best_state); pol.eval()
    return pol, hist


def save_zone_model(pol, path): torch.save(pol.state_dict(), path)
def load_zone_model(path):
    pol = Policy(fin=FIN, de=DE); pol.load_state_dict(torch.load(path)); return pol.eval()
