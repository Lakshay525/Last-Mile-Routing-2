"""Figures for the report from the final CSVs in results/ (run from the repo root)."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lmr.analysis import summary, paired

R = Path("results"); OUT = R / "figures"; OUT.mkdir(exist_ok=True)
LNS, SA = "lns:i=5000:kmax=2", "sa:i=5000000:t0=0.5"
ZL = ":w=3:ii=1000:wi=1500:kmax=2"; G20 = "gnn:beam=20"
df = pd.concat([pd.read_csv(R / f"{n}.csv") for n in ("final_f1_test", "final_f5_gnn_order", "final_f2_ortools")]).drop_duplicates(["route", "method"])
s = summary(df, ["gap_vs_driver", "zone_sw", "official_score"])
g = lambda m, c: s.loc[m, c]

curves = {  # label -> methods in order of increasing lambda / rho
    "plain LNS (approach 2), lambda = 0, 15, 30, 60, 120": [f"{LNS}:l={l}" for l in (0, 15, 30, 60, 120)],
    "zonal LNS, zone-level routing order, lambda = 0, 60": [f"zlns:order=ztsp{ZL}:l={l}" for l in (0, 60)],
    "zonal LNS, reference-route order, lambda = 0, 60": [f"zlns:order=ref{ZL}:l={l}" for l in (0, 60)],
    "zonal LNS, reference order + global refinement, rho = 30, 100, 300": [f"zlns:order=ref{ZL}:g=5000:rho={r}" for r in (30, 100, 300)],
    "zonal LNS, driver's order (oracle), lambda = 0, 60": [f"zlns:order=driver{ZL}:l={l}" for l in (0, 60)],
}
points = [("driver", "driver", "*"), ("nn", "nearest neighbour", "s"), ("ortools:t=10", "OR-Tools 10 s (lambda=0)", "D"), ("ortools:t=10:l=60", "OR-Tools 10 s (lambda=60)", "d"),
          (f"{SA}:l=0", "global SA (lambda=0)", "^"), (f"{SA}:l=60", "global SA (lambda=60)", "v"), (f"zlns:order={G20}{ZL}:l=60", "zonal LNS, GNN+reference order, lambda=60", "P")]

# 1. official score vs travel time (the trade-off between being fast and looking like the driver)
fig, ax = plt.subplots(figsize=(8.5, 6))
for lab, ms in curves.items():
    ms = [m for m in ms if m in s.index]
    ax.plot([g(m, "gap_vs_driver") * 100 for m in ms], [g(m, "official_score") for m in ms], "o-", label=lab)
for m, lab, mk in points:
    if m in s.index: ax.scatter(g(m, "gap_vs_driver") * 100, g(m, "official_score"), marker=mk, s=80, c="k", zorder=5, label=lab)
ax.set_xlabel("travel time vs driver, % (lower = faster)"); ax.set_ylabel("official challenge score (lower = closer to the driver)")
ax.set_title("Speed against driver-likeness, 207 test routes"); ax.legend(fontsize=6.5); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / "score_vs_time.png", dpi=150); plt.close(fig)

# 2. zone changes vs travel time
fig, ax = plt.subplots(figsize=(8.5, 6))
for lab, ms in curves.items():
    ms = [m for m in ms if m in s.index]
    ax.plot([g(m, "zone_sw") for m in ms], [g(m, "gap_vs_driver") * 100 for m in ms], "o-", label=lab)
for m, lab, mk in points:
    if m in s.index: ax.scatter(g(m, "zone_sw"), g(m, "gap_vs_driver") * 100, marker=mk, s=80, c="k", zorder=5, label=lab)
ax.set_xlabel("zone changes per route (lower = tidier)"); ax.set_ylabel("travel time vs driver, % (lower = faster)")
ax.set_title("Travel time against tidiness, 207 test routes"); ax.legend(fontsize=6.5); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / "tradeoff_zone_changes.png", dpi=150); plt.close(fig)

# 3. paired differences vs OR-Tools
pairs = [(f"{LNS}:l=0", "ortools:t=10", "plain LNS"), (f"{SA}:l=0", "ortools:t=10", "global SA"), (f"zlns:order=ztsp{ZL}:l=0", "ortools:t=10", "zonal LNS (zone-level routing order)"),
         (f"zlns:order=ref{ZL}:l=0", "ortools:t=10", "zonal LNS (reference order)"), (f"{LNS}:l=60", "ortools:t=10:l=60", "plain LNS, lambda=60"),
         (f"zlns:order=ref{ZL}:l=60", "ortools:t=10:l=60", "zonal LNS (reference order), lambda=60")]
rows = [(lab, paired(df, a, b)) for a, b, lab in pairs]
fig, ax = plt.subplots(figsize=(8, 3.6)); y = np.arange(len(rows))
ax.errorbar([r["mean_pts"] for _, r in rows], y, xerr=[r["ci95"] for _, r in rows], fmt="o", capsize=3)
ax.axvline(0, c="k", lw=1); ax.set_yticks(y); ax.set_yticklabels([l for l, _ in rows], fontsize=8); ax.invert_yaxis()
ax.set_xlabel("percentage points of driver travel time (negative = less travel time than OR-Tools)", fontsize=8)
ax.set_title("Paired difference vs OR-Tools 10 s (mean +- 95% CI, 207 routes)", fontsize=10); fig.tight_layout()
fig.savefig(OUT / "vs_ortools.png", dpi=150); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
