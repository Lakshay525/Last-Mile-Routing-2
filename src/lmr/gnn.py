import copy, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import kendalltau


def featurize(r):
    n = len(r.T); eye = np.eye(n, dtype=bool); Tn = r.T / r.T[~eye].mean()
    off = np.where(eye, np.nan, Tn)
    xy = (r.xy - r.xy.mean(0)) / (r.xy.std(0).mean() + 1e-9)
    stats = [np.nanmean(off, 1), np.nanmin(off, 1), np.nanmax(off, 1),
             np.nanmean(off, 0), np.nanmin(off, 0), np.nanmax(off, 0)]
    zsz = np.array([(r.zone == z).mean() if z >= 0 else 0 for z in r.zone])
    st = np.zeros(n); st[0] = 1
    x = np.column_stack([xy, *stats, np.linalg.norm(xy, axis=1), zsz, st])          # 11 features
    same = (r.zone[:, None] == r.zone[None, :]).astype(float)
    e = np.stack([Tn, Tn.T, same], -1)                                                # asymmetric edge feats
    t = lambda a: torch.tensor(a, dtype=torch.float32)[None]
    return t(x), t(e), t(Tn)


class EdgeGATv2(nn.Module):
    def __init__(s, d, de, heads=4):
        super().__init__(); s.h, s.dk = heads, d // heads
        s.Wl, s.Wr = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        s.We = nn.Linear(de, d, bias=False); s.a = nn.Parameter(torch.randn(heads, s.dk) * 0.1)
        s.out, s.norm, s.drop = nn.Linear(d, d), nn.LayerNorm(d), nn.Dropout(0.1)

    def forward(s, h, e):
        B, n, d = h.shape
        l = s.Wl(h).view(B, n, 1, s.h, s.dk); r = s.Wr(h).view(B, 1, n, s.h, s.dk)
        g = F.leaky_relu(l + r + s.We(e).view(B, n, n, s.h, s.dk), 0.2)
        att = (g * s.a).sum(-1).softmax(2).unsqueeze(-1)
        m = (att * r).sum(2).reshape(B, n, d)
        return s.norm(h + s.drop(F.elu(s.out(m))))


class PtrDecoder(nn.Module):
    def __init__(s, d):
        super().__init__(); s.d = d; s.gru = nn.GRUCell(2 * d, d)
        s.Wq, s.Wk = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        s.beta = nn.Parameter(torch.tensor(1.0))

    def forward(s, H, Tn, mode="sample", target=None):
        B, n, d = H.shape; ar = torch.arange(B); hg = H.mean(1); k = s.Wk(H)
        state, cur = hg.clone(), torch.zeros(B, dtype=torch.long)
        visited = torch.zeros(B, n, dtype=torch.bool); visited[:, 0] = True
        seq, logp, hits = [cur], 0, []
        for t in range(n - 1):
            state = s.gru(torch.cat([H[ar, cur], hg], -1), state)
            u = 10 * torch.tanh((s.Wq(state)[:, None] * k).sum(-1) / d ** 0.5) - s.beta * Tn[ar, cur]
            lp = u.masked_fill(visited, float("-inf")).log_softmax(-1)
            nxt = target[:, t + 1] if target is not None else (
                  lp.argmax(-1) if mode == "greedy" else torch.multinomial(lp.exp(), 1).squeeze(1))
            if target is not None: hits.append(lp.argmax(-1) == nxt)
            logp = logp + lp[ar, nxt]; visited = visited.scatter(1, nxt[:, None], True)
            seq.append(nxt); cur = nxt
        s.last_hits = torch.stack(hits, 1) if hits else None      # teacher-forced: was the model's top choice the target?
        return torch.stack(seq, 1), logp


def _beam_decode(dec, H, Tn, width):
    """Beam search over whole sequences for a PtrDecoder, batch size 1. Returns the sequence with the highest
    total log-probability among `width` partial sequences kept at each step."""
    B, n, d = H.shape; hg = H.mean(1); k = dec.Wk(H)
    beams = [(0.0, [0], hg.clone(), torch.zeros(1, n, dtype=torch.bool))]
    beams[0][3][0, 0] = True
    for t in range(n - 1):
        cand = []
        for lp0, seq, state, visited in beams:
            cur = seq[-1]
            state = dec.gru(torch.cat([H[:, cur], hg], -1), state)
            u = 10 * torch.tanh((dec.Wq(state)[:, None] * k).sum(-1) / d ** 0.5) - dec.beta * Tn[:, cur]
            lp = u.masked_fill(visited, float("-inf")).log_softmax(-1)[0]
            top = torch.topk(lp, min(width, int((~visited[0]).sum())))
            for v, j in zip(top.values.tolist(), top.indices.tolist()):
                nv = visited.clone(); nv[0, j] = True
                cand.append((lp0 + v, seq + [j], state, nv))
        cand.sort(key=lambda c: -c[0]); beams = cand[:width]
    return beams[0][1], beams[0][0]


class Policy(nn.Module):
    def __init__(s, fin=11, d=64, layers=3, de=3):
        super().__init__(); s.inp = nn.Linear(fin, d)
        s.gnn = nn.ModuleList(EdgeGATv2(d, de) for _ in range(layers)); s.dec = PtrDecoder(d)

    def forward(s, x, e, Tn, mode="sample", target=None):
        h = s.inp(x)
        for l in s.gnn: h = l(h, e)
        return s.dec(h, Tn, mode, target)


def tour_cost(Tn, seq):
    b = torch.arange(len(seq))[:, None]; return Tn[b, seq[:, :-1], seq[:, 1:]].sum(1)


def train(routes, val_routes, epochs=20, lr=1e-4, bs=8, imitation_epochs=3,
          n_samples=8, rl_lr=2e-5, patience=3):
    """Phase 1 (imitation): copy the driver's next stop.
    Phase 2 (REINFORCE): K sampled tours per instance, baseline = their mean, advantage normalised.
    Each epoch logs validation route cost AND similarity to the driver.
    The returned model is the end-of-imitation model, or a later one if RL beats its cost.
    Early stopping only counts during the RL phase. Returns (policy, history)."""
    from .solvers import nearest_neighbour
    pol = Policy(); opt = torch.optim.Adam(pol.parameters(), lr)
    data = [featurize(r) for r in routes]; val = [featurize(r) for r in val_routes]

    # reference points in the same units
    nn_c = np.mean([tour_cost(Tn, torch.tensor([nearest_neighbour(Tn[0].numpy())])).item() for _, _, Tn in val])
    drv_c = np.mean([tour_cost(Tn, torch.arange(Tn.shape[1])[None]).item() for _, _, Tn in val])
    print(f"reference on val: nearest-neighbour {nn_c:.2f}  driver {drv_c:.2f}")

    history, best, best_state, bad = [], float("inf"), None, 0
    for ep in range(epochs):
        random.shuffle(data); pol.train()
        if ep == imitation_epochs:                      # gentler learning rate for the RL phase
            for g in opt.param_groups: g["lr"] = rl_lr
        for i in range(0, len(data), bs):
            loss = 0
            for x, e, Tn in data[i:i + bs]:
                if ep < imitation_epochs:
                    tgt = torch.arange(x.shape[1])[None]; _, lp = pol(x, e, Tn, target=tgt); loss = loss - lp.mean()
                else:
                    X, E, TN = (x.expand(n_samples, -1, -1), e.expand(n_samples, -1, -1, -1),
                                Tn.expand(n_samples, -1, -1))
                    seq, lp = pol(X, E, TN, "sample")
                    c = tour_cost(TN, seq)
                    adv = ((c - c.mean()) / (c.std() + 1e-8)).detach()
                    loss = loss + (adv * lp).mean()
            opt.zero_grad(); (loss / bs).backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()

        pol.eval()
        with torch.no_grad():
            outs = [pol(x, e, Tn, "greedy")[0][0] for x, e, Tn in val]
        pc = float(np.mean([tour_cost(Tn, o[None]).item() for (x, e, Tn), o in zip(val, outs)]))
        sim = float(np.mean([kendalltau(np.argsort(o.numpy()), np.arange(len(o)))[0] for o in outs]))
        history.append(dict(epoch=ep, val_policy=pc, val_sim=sim, nn_ref=float(nn_c), driver_ref=float(drv_c)))

        if ep < imitation_epochs - 1:
            pass                                        # still imitating: no selection, no stopping
        elif ep == imitation_epochs - 1:                # end of imitation = the starting point to beat
            best, best_state, bad = pc, copy.deepcopy(pol.state_dict()), 0
        elif pc < best:
            best, best_state, bad = pc, copy.deepcopy(pol.state_dict()), 0
        else:
            bad += 1
        print(f"epoch {ep}  val cost {pc:.2f}  val similarity {sim:.3f}  (best {best:.2f})")
        if ep >= imitation_epochs and bad >= patience:
            print(f"early stop: no improvement for {patience} epochs"); break

    if best_state is None:                              # fewer epochs than imitation_epochs
        best_state = copy.deepcopy(pol.state_dict())
    pol.load_state_dict(best_state)
    return pol, history


def gnn_route(pol, r):
    x, e, Tn = featurize(r)
    with torch.no_grad(): return pol(x, e, Tn, "greedy")[0][0].tolist()


def save_model(pol, path): torch.save(pol.state_dict(), path)


def load_model(path):
    pol = Policy(); pol.load_state_dict(torch.load(path)); return pol.eval()


def gnn_solvers(pol):
    from .solvers import simulated_annealing, lns
    from . import config
    return {
        "gnn":     lambda r: gnn_route(pol, r),
        "gnn_sa":  lambda r: simulated_annealing(r.T, init=gnn_route(pol, r), iters=config.SA_ITERS)[0],
        "gnn_lns": lambda r: lns(r.T, r.zone, init=gnn_route(pol, r), iters=config.LNS_ITERS)[0],
    }