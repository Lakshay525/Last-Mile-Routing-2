import copy, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

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
        seq, logp = [cur], 0
        for t in range(n - 1):
            state = s.gru(torch.cat([H[ar, cur], hg], -1), state)
            u = 10 * torch.tanh((s.Wq(state)[:, None] * k).sum(-1) / d ** 0.5) - s.beta * Tn[ar, cur]
            lp = u.masked_fill(visited, float("-inf")).log_softmax(-1)
            nxt = target[:, t + 1] if target is not None else (
                  lp.argmax(-1) if mode == "greedy" else torch.multinomial(lp.exp(), 1).squeeze(1))
            logp = logp + lp[ar, nxt]; visited = visited.scatter(1, nxt[:, None], True)
            seq.append(nxt); cur = nxt
        return torch.stack(seq, 1), logp

class Policy(nn.Module):
    def __init__(s, fin=11, d=64, layers=3):
        super().__init__(); s.inp = nn.Linear(fin, d)
        s.gnn = nn.ModuleList(EdgeGATv2(d, 3) for _ in range(layers)); s.dec = PtrDecoder(d)
    def forward(s, x, e, Tn, mode="sample", target=None):
        h = s.inp(x)
        for l in s.gnn: h = l(h, e)
        return s.dec(h, Tn, mode, target)

def tour_cost(Tn, seq):
    b = torch.arange(len(seq))[:, None]; return Tn[b, seq[:, :-1], seq[:, 1:]].sum(1)

def train(routes, val_routes, epochs=20, lr=1e-4, bs=8, imitation_epochs=3):
    """Imitation warm-up (copy the driver), then REINFORCE with a greedy-rollout baseline.
    Returns (policy, history)."""
    pol = Policy(); base = copy.deepcopy(pol).eval(); opt = torch.optim.Adam(pol.parameters(), lr)
    data = [featurize(r) for r in routes]; val = [featurize(r) for r in val_routes]
    history = []
    for ep in range(epochs):
        random.shuffle(data); pol.train()
        for i in range(0, len(data), bs):
            loss = 0
            for x, e, Tn in data[i:i + bs]:
                if ep < imitation_epochs:
                    tgt = torch.arange(x.shape[1])[None]; _, lp = pol(x, e, Tn, target=tgt); loss = loss - lp.mean()
                else:
                    seq, lp = pol(x, e, Tn, "sample")
                    with torch.no_grad(): bseq, _ = base(x, e, Tn, "greedy")
                    loss = loss + ((tour_cost(Tn, seq) - tour_cost(Tn, bseq)).detach() * lp).mean()
            opt.zero_grad(); (loss / bs).backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
        pol.eval()
        with torch.no_grad():
            pc = float(np.mean([tour_cost(Tn, pol(x, e, Tn, "greedy")[0]).item() for x, e, Tn in val]))
            bc = float(np.mean([tour_cost(Tn, base(x, e, Tn, "greedy")[0]).item() for x, e, Tn in val]))
        if pc < bc: base.load_state_dict(pol.state_dict())
        history.append(dict(epoch=ep, val_policy=pc, val_baseline=bc))
        print(f"epoch {ep}  val greedy cost  policy {pc:.2f}  baseline {bc:.2f}")
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