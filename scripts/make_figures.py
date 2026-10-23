"""Figures for the report, from the CSVs in results/ (run from the repo root)."""
import re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lmr.analysis import summary, paired

R = Path("results"); OUT = R / "figures"; OUT.mkdir(exist_ok=True)
names = ["exp10_no_gnn", "exp11_penalty", "exp12_gnn_order", "exp13_ortools"]
df = pd.concat([pd.read_csv(R / f"{n}.csv") for n in names if (R / f"{n}.csv").exists()])
s = summary(df); s["gap_pct"] = s.gap_vs_driver * 100
lam = lambda m: int(re.search(r":l=(\d+)", m).group(1)) if ":l=" in m else 0
fam = lambda m: re.sub(r":l=\d+", "", m)
s["lam"] = [lam(m) for m in s.index]; s["family"] = [fam(m) for m in s.index]

# 1. trade-off: travel time saved vs zone changes
curves = {"lns:i=1500": "plain LNS", "zlns:order=ztsp:w=3": "zone LNS, zone-level routing order, w=3",
          "zlns:order=gnn:w=3": "zone LNS, GNN order, w=3", "zlns:order=greedy:w=2": "zone LNS, greedy order, w=2",
          "zlns:order=driver:w=2": "zone LNS, driver's zone order (oracle), w=2"}
fig, ax = plt.subplots(figsize=(8, 5.5))
for f, lab in curves.items():
    g = s[s.family == f].sort_values("lam")
    if f == "zlns:order=gnn:w=3": g = s[s.index.str.startswith("zlns:order=gnn:w=3")].sort_values("lam")
    if len(g):
        ax.plot(g.zone_sw, g.gap_pct, "o-", label=lab)
        for _, r in g.iterrows(): ax.annotate(f"λ={int(r.lam)}", (r.zone_sw, r.gap_pct), fontsize=6, xytext=(3, 3), textcoords="offset points")
for m, lab, mk in [("driver", "driver", "*"), ("nn", "nearest neighbour", "s"), ("ortools:t=10", "OR-Tools 10 s (λ=0)", "D"),
                   ("ortools:t=10:l=60", "OR-Tools 10 s (λ=60)", "d")]:
    if m in s.index: ax.scatter(s.loc[m, "zone_sw"], s.loc[m, "gap_pct"], marker=mk, s=90, c="k", zorder=5, label=lab)
ax.set_xlabel("zone changes per route (lower = tidier)"); ax.set_ylabel("travel time vs driver, % (lower = faster)")
ax.set_title("Travel time vs tidiness, 207 test routes\n(λ = penalty in seconds per zone change)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "tradeoff.png", dpi=150); plt.close(fig)

# 2. paired differences versus OR-Tools (percentage points of the driver's travel time; negative = better than OR-Tools)
pairs = [("lns:i=500", "ortools:t=10"), ("lns:i=1500", "ortools:t=10"), ("lns:i=15000", "ortools:t=10"),
         ("lns:i=1500:l=60", "ortools:t=10:l=60"), ("zlns:order=ztsp:w=3", "ortools:t=10"),
         ("zlns:order=gnn:w=3:l=0", "ortools:t=10"), ("zlns:order=ztsp:w=3:l=60", "ortools:t=10:l=60")]
rows = [paired(df, a, b) for a, b in pairs if a in set(df.method) and b in set(df.method)]
fig, ax = plt.subplots(figsize=(8, 3.8))
y = np.arange(len(rows))
ax.errorbar([r["mean_pts"] for r in rows], y, xerr=[r["ci95"] for r in rows], fmt="o", capsize=3)
ax.axvline(0, c="k", lw=1); ax.set_yticks(y); ax.set_yticklabels([f'{r["a"]}  vs  {r["b"]}' for r in rows], fontsize=7)
ax.invert_yaxis(); ax.set_xlabel("percentage points of driver travel time (negative = first method better)", fontsize=8)
ax.set_title("Paired difference vs OR-Tools 10 s (mean ± 95% CI, 207 routes)", fontsize=10); fig.tight_layout()
fig.savefig(OUT / "vs_ortools.png", dpi=150); plt.close(fig)
print("wrote", [p.name for p in OUT.glob("*.png")])
