import math

class MethodBandit:
    """UCB1 over methods; context = route-size bucket. reward e.g. -gap_vs_driver."""
    def __init__(s, names): s.n, s.m, s.names = {}, {}, names
    def pick(s, ctx):
        tot = sum(s.n.get((ctx, k), 0) for k in s.names) + 1
        return max(s.names, key=lambda k: s.m.get((ctx, k), 0) +
                   (math.sqrt(2 * math.log(tot) / s.n[(ctx, k)]) if s.n.get((ctx, k)) else 1e9))
    def update(s, ctx, k, reward):
        c = (ctx, k); s.n[c] = s.n.get(c, 0) + 1; s.m[c] = s.m.get(c, 0) + (reward - s.m.get(c, 0)) / s.n[c]

def size_bucket(n): return "small" if n < 100 else "medium" if n < 160 else "large"