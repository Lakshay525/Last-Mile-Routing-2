"""Figures and numbers for the report (paper/main.tex). Run from the repo root:  python paper/make_paper_figures.py
Everything is read from the CSV files in results/ (or recomputed from the data for the example route); nothing is typed by hand."""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R = Path("results"); OUT = Path("paper/figures"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 120})
(OUT / "preview").mkdir(exist_ok=True)
_savefig = matplotlib.figure.Figure.savefig


def _save_with_preview(self, fname, *a, **k):          # every figure is also written as a PNG preview
    _savefig(self, fname, *a, **k); _savefig(self, OUT / "preview" / (Path(fname).stem + ".png"), dpi=130)


matplotlib.figure.Figure.savefig = _save_with_preview
LNS, SA = "lns:i=5000:kmax=2", "sa:i=5000000:t0=0.5"
ZL ="w=3:ii=1000:wi=1500:kmax=2"; ZS = "w=3:ii=50000:wi=300000:t0=0.5"

# ------------------------------------------------------------------ data
seed0 = pd.read_csv(R / "final_f1_test.csv").assign(seed=0)
seeds = pd.concat([seed0] + [pd.read_csv(f).assign(seed=int(f.split("seed")[-1][:-4])) for f in sorted(glob.glob(str(R / "final_f8_seed*.csv")))])
ort = pd.read_csv(R / "final_f2_ortools.csv")
METHODS = {  # label -> (spec prefix, colour, marker)
    "plain LNS": (LNS, "#1b6ca8", "o"),
    "plain SA": (SA, "#7f7f7f", "s"),
    "zonal LNS, order by zone routing": (f"zlns:order=ztsp:{ZL}", "#e08214", "^"),
    "zonal LNS, order from past routes": (f"zlns:order=ref:{ZL}", "#b2182b", "D"),
    "zonal SA, order by zone routing": (f"zsa:order=ztsp:{ZS}", "#fdb863", "v"),
    "zonal SA, order from past routes": (f"zsa:order=ref:{ZS}", "#762a83", "P"),
}


def per_seed(spec):
    d = seeds[seeds.method == spec].groupby("seed").agg(t=("gap_vs_driver", lambda x: 100 * x.mean()), z=("zone_sw", "mean"), s=("official_score", "mean"))
    return d


def stats_table():
    rows = []
    for lam in (0, 60):
        o = ort[ort.method == ("ortools:t=10" if lam == 0 else "ortools:t=10:l=60")]
        rows.append(dict(lam=lam, method="OR-Tools (10 s)", t=100 * o.gap_vs_driver.mean(), t_sd=np.nan, z=o.zone_sw.mean(), z_sd=np.nan, s=o.official_score.mean(), s_sd=np.nan, seeds=1))
        for lab, (spec, _, _) in METHODS.items():
            d = per_seed(f"{spec}:l={lam}")
            rows.append(dict(lam=lam, method=lab, t=d.t.mean(), t_sd=d.t.std(), z=d.z.mean(), z_sd=d.z.std(), s=d.s.mean(), s_sd=d.s.std(), seeds=len(d)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Fig 1: trade-off between time and driver-likeness
def fig_tradeoff(tab):
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter([0], [0], marker="*", s=140, c="k", zorder=5, label="driver (reference)")
    cols = {"OR-Tools (10 s)": ("#2ca02c", "X")}
    cols.update({k: (v[1], v[2]) for k, v in METHODS.items()})
    for lab, (c, mk) in cols.items():
        d = tab[tab.method == lab].sort_values("lam")
        ax.plot(d.t, d.s, "-", c=c, lw=1, alpha=.7)
        ax.scatter(d.t[d.lam == 0], d.s[d.lam == 0], marker=mk, s=45, facecolors="none", edgecolors=c, linewidths=1.5, zorder=4)
        ax.scatter(d.t[d.lam == 60], d.s[d.lam == 60], marker=mk, s=45, c=c, zorder=4, label=lab)
    ax.set_xlabel("travel time relative to the driver (%)   [left = faster]"); ax.set_ylabel("official score   [lower = more like the driver]")
    ax.text(0.01, 0.99, "hollow marker: $\\lambda=0$, filled marker: $\\lambda=60$ s", transform=ax.transAxes, va="top", fontsize=8)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2); ax.invert_xaxis()
    fig.tight_layout(); fig.savefig(OUT / "fig_tradeoff.pdf"); plt.close(fig)


# ------------------------------------------------------------------ Fig 2: effect of lambda (plain LNS)
def fig_lambda():
    lams = [0, 15, 30, 60, 120]; d = seed0
    g = [100 * d[d.method == f"{LNS}:l={l}"].gap_vs_driver.mean() for l in lams]
    z = [d[d.method == f"{LNS}:l={l}"].zone_sw.mean() for l in lams]
    s = [d[d.method == f"{LNS}:l={l}"].official_score.mean() for l in lams]
    zd = d[d.method == "driver"].zone_sw.mean()
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for a, y, t in zip(ax, (g, z, s), ("travel time vs driver (%)", "zone changes per route", "official score")):
        a.plot(lams, y, "o-", c="#1b6ca8"); a.set_xlabel("$\\lambda$ (s per zone change)"); a.set_title(t, fontsize=9)
    ax[0].axhline(0, c="k", lw=.8); ax[1].axhline(zd, c="k", ls="--", lw=.8); ax[1].text(lams[-1], zd + .6, "driver", ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_lambda.pdf"); plt.close(fig)


# ------------------------------------------------------------------ Fig 3: run time
def fig_time():
    d = pd.read_csv(R / "final_f4_timing.csv")
    rows = [("OR-Tools, 10 s limit", "ortools:t=10"), ("OR-Tools, 2 s limit", "ortools:t=2"), ("plain LNS", LNS), ("plain SA", SA),
            ("zonal SA (zone routing order)", f"zsa:order=ztsp:{ZS}"), ("zonal LNS (zone routing order)", f"zlns:order=ztsp:{ZL}"), ("zonal LNS (past-routes order)", f"zlns:order=ref:{ZL}")]
    v = [d[d.method == m].secs.mean() for _, m in rows]
    fig, ax = plt.subplots(figsize=(6, 2.6)); y = np.arange(len(rows))
    ax.barh(y, v, color="#1b6ca8"); ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels([l for l, _ in rows], fontsize=8); ax.invert_yaxis()
    for yi, vi in zip(y, v): ax.text(vi * 1.08, yi, f"{vi:.2f} s", va="center", fontsize=8)
    ax.set_xlim(0.05, 40); ax.set_xlabel("mean run time per route (s, log scale, one process)"); ax.grid(axis="y", alpha=0)
    fig.tight_layout(); fig.savefig(OUT / "fig_time.pdf"); plt.close(fig)


# ------------------------------------------------------------------ Fig 4: window overlap ablation
def fig_ablation():
    a = pd.read_csv(R / "final_f6_ablation.csv"); fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for order, lab, c in (("ref", "order from past routes", "#b2182b"), ("ztsp", "order by zone routing", "#e08214")):
        ws = [1, 2, 3, 4, 5]; sub = [a[a.method == f"zlns:order={order}:w={w}:ii=1000:wi=1500:kmax=2:l=60"] for w in ws]
        ax[0].plot(ws, [100 * s.gap_vs_driver.mean() for s in sub], "o-", c=c, label=lab); ax[1].plot(ws, [s.official_score.mean() for s in sub], "o-", c=c, label=lab)
    ax[0].set_ylabel("travel time vs driver (%)"); ax[1].set_ylabel("official score")
    for x in ax: x.set_xlabel("window width $w$ (zones)"); x.set_xticks([1, 2, 3, 4, 5])
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "fig_ablation.pdf"); plt.close(fig)


# ------------------------------------------------------------------ Fig 5: dataset
def fig_data():
    from lmr import load_routes, split
    Rt = load_routes(6112, 0); n = np.array([len(r.T) - 1 for r in Rt]); z = np.array([len(set(r.zone[1:].tolist())) for r in Rt])
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3))
    ax[0].hist(n, bins=30, color="#1b6ca8"); ax[0].set_xlabel("stops per route"); ax[0].set_ylabel("routes")
    ax[1].hist(z, bins=range(0, 50, 2), color="#e08214"); ax[1].set_xlabel("zones per route")
    fig.tight_layout(); fig.savefig(OUT / "fig_data.pdf"); plt.close(fig)
    tr, va, te = split(Rt); P = load_routes(1000, 0); t2 = split(P)[2]
    q = pd.Series([r.score for r in Rt]).value_counts()
    info = dict(routes=len(Rt), stations=len({r.station for r in Rt}), stops_mean=n.mean(), stops_min=n.min(), stops_max=n.max(), zones_mean=z.mean(), zones_min=z.min(), zones_max=z.max(),
                high=q["High"], medium=q["Medium"], low=q["Low"], train=len(tr), val=len(va), test_all=len(te), test_used=len(t2),
                test_stops=np.mean([len(r.T) - 1 for r in t2]), test_zones=np.mean([len(set(r.zone[1:].tolist())) for r in t2]))
    pd.Series(info).to_csv(OUT / "dataset_numbers.csv", header=False); print(pd.Series(info).round(2).to_string())


# ------------------------------------------------------------------ Fig 6: one example route
def fig_routes():
    from lmr import load_routes, split
    from lmr.methods import run_spec
    te = split(load_routes(1000, 0))[2]; te = [r for r in te if r.score == "High"]
    n = np.array([len(r.T) - 1 for r in te]); r = te[int(np.argmin(np.abs(n - np.median(n))))]
    panels = [("driver", "driver", "the driver's route"), (f"{LNS}:l=0", None, "plain LNS, $\\lambda=0$"), (f"{LNS}:l=60", None, "plain LNS, $\\lambda=60$"),
              (f"zlns:order=ref:{ZL}:l=60", None, "zonal LNS (past-routes order), $\\lambda=60$")]
    zid = {z: i for i, z in enumerate(sorted(set(r.zone[1:].tolist())))}; cmap = plt.get_cmap("tab20"); col = [cmap(zid[z] % 20) if i else (0, 0, 0, 1) for i, z in enumerate(r.zone)]
    T = r.T; drv = T[np.arange(len(T) - 1), np.arange(1, len(T))].sum(); fig, axs = plt.subplots(2, 2, figsize=(6.6, 6.2))
    for ax, (spec, _, title) in zip(axs.ravel(), panels):
        p = run_spec(spec, r, 0); assert sorted(p) == list(range(len(T))) and p[0] == 0
        jt = T[p[:-1], p[1:]].sum(); zc = int((r.zone[p[1:-1]] != r.zone[p[2:]]).sum())
        xy = r.xy[p[1:]]; ax.plot(xy[:, 1], xy[:, 0], "-", c="#555555", lw=.7, alpha=.8, zorder=1)      # the depot is far away and not drawn
        ax.scatter(r.xy[1:, 1], r.xy[1:, 0], c=[col[i] for i in range(1, len(col))], s=14, zorder=2, edgecolors="none")
        ax.scatter([xy[0, 1]], [xy[0, 0]], marker="*", s=130, c="k", zorder=3)               # first stop of the route
        pad = 0.04 * (r.xy[1:].max(0) - r.xy[1:].min(0)); lo, hi = r.xy[1:].min(0) - pad, r.xy[1:].max(0) + pad
        ax.set_xlim(lo[1], hi[1]); ax.set_ylim(lo[0], hi[0])
        ax.set_title(f"{title}\n{jt:.0f} s ({100 * (jt / drv - 1):+.1f}% vs driver), {zc} zone changes", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_routes.pdf"); plt.close(fig); print("example route", r.rid, "stops", len(T) - 1, "zones", len(zid))


# ------------------------------------------------------------------ Fig 7: pipeline schematic
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(6.8, 3.2)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 48)
    def box(x, y, w, h, txt, fc="#dbe9f6"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc=fc, ec="#1b6ca8", lw=1)); ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7.5)
    def arrow(x0, y0, x1, y1): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1, color="#333"))
    box(1, 33, 20, 11, "Input\nstops, zone IDs,\ntravel-time matrix $T$", "#eeeeee")
    box(28, 33, 20, 11, "Step 1\nchoose the order\nof the zones")
    box(55, 33, 20, 11, "Step 2\nsolve each zone\n(LNS or SA)")
    box(79, 33, 20, 11, "Step 3\nre-optimise windows\nof $w=3$ zones")
    for a, b in ((21, 28), (48, 55), (75, 79)): arrow(a + .5, 38.5, b - .5, 38.5)
    ax.text(1, 27.5, "Step 3 in detail: windows of three neighbouring zones overlap, so stops near a zone border can still move", fontsize=7.5, style="italic")
    cs = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462"]
    for i, c in enumerate(cs): ax.add_patch(FancyBboxPatch((6 + i * 15, 12), 13, 7, boxstyle="round,pad=0.2", fc=c, ec="#555", lw=.8)); ax.text(12.5 + i * 15, 15.5, f"zone {i + 1}", ha="center", va="center", fontsize=7)
    for k, (x0, yy, c) in enumerate(((6, 8.5, "#1b6ca8"), (21, 5.5, "#b2182b"), (36, 2.5, "#2ca02c"))):
        ax.plot([x0, x0 + 43], [yy, yy], "-", c=c, lw=3); ax.text(x0 + 45, yy, f"window {k + 1}: zones {k + 1}-{k + 3}", va="center", fontsize=7)
    ax.text(1, 21, "route (order fixed by step 1)", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig_pipeline.pdf"); plt.close(fig)


# ------------------------------------------------------------------ LaTeX tables (read by main.tex with \input)
TAB = Path("paper/tables"); TAB.mkdir(exist_ok=True)


def pm(m, s, d):
    return f"${m:.{d}f}$" if (s is None or np.isnan(s)) else f"${m:.{d}f}\\pm{s:.{d}f}$"


def table_main(tab):
    L = []
    for lam in (0, 60):
        L.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{'$\\lambda=0$ (travel time only)' if lam == 0 else '$\\lambda=60$ s (balanced)'}}}}}\\\\")
        for _, r in tab[tab.lam == lam].iterrows():
            L.append(f"{r.method} & {pm(r.t, r.t_sd, 2)} & {pm(r.z, r.z_sd, 1)} & {pm(r.s, r.s_sd, 4)} \\\\")
        L.append("\\midrule" if lam == 0 else "")
    (TAB / "main.tex").write_text("\n".join(L), encoding="utf-8")


def _pair(df, col, a, b, scale=1.0):
    w = df.pivot(index="route", columns="method", values=col) * scale; d = (w[a] - w[b]).dropna()
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d)), 100 * (d < 0).mean(), len(d)


def table_paired():
    df = pd.concat([seed0, ort]).drop_duplicates(["route", "method"])
    P = lambda s: s
    ZLx, ZSx = lambda o, l: f"zlns:order={o}:{ZL}:l={l}", lambda o, l: f"zsa:order={o}:{ZS}:l={l}"
    rows = [("travel time (points)", "gap_vs_driver", 100, [
        ("$\\lambda=0$: plain LNS vs OR-Tools", f"{LNS}:l=0", "ortools:t=10"), ("$\\lambda=0$: plain SA vs OR-Tools", f"{SA}:l=0", "ortools:t=10"),
        ("$\\lambda=0$: plain SA vs plain LNS", f"{SA}:l=0", f"{LNS}:l=0"), ("$\\lambda=0$: zonal LNS (zone-routing order) vs OR-Tools", ZLx("ztsp", 0), "ortools:t=10"),
        ("$\\lambda=0$: zonal LNS (zone-routing order) vs plain LNS", ZLx("ztsp", 0), f"{LNS}:l=0"), ("$\\lambda=0$: zonal SA vs zonal LNS (zone-routing order)", ZSx("ztsp", 0), ZLx("ztsp", 0)),
        ("$\\lambda=60$: plain LNS vs OR-Tools", f"{LNS}:l=60", "ortools:t=10:l=60"), ("$\\lambda=60$: zonal LNS (past-routes order) vs OR-Tools", ZLx("ref", 60), "ortools:t=10:l=60"),
        ("$\\lambda=60$: zonal LNS (past-routes order) vs plain LNS", ZLx("ref", 60), f"{LNS}:l=60")]),
        ("official score", "official_score", 1, [
            ("$\\lambda=60$: zonal LNS (past-routes order) vs OR-Tools", ZLx("ref", 60), "ortools:t=10:l=60"), ("$\\lambda=60$: zonal LNS (past-routes order) vs plain LNS", ZLx("ref", 60), f"{LNS}:l=60"),
            ("$\\lambda=60$: zonal LNS, past-routes vs zone-routing order", ZLx("ref", 60), ZLx("ztsp", 60))])]
    L = []
    for name, col, sc, items in rows:
        L.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{name}}}}}\\\\")
        for lab, a, b in items:
            m, ci, share, n = _pair(df, col, a, b, sc); d = 2 if col == "gap_vs_driver" else 4
            L.append(f"{lab} & ${m:+.{d}f}$ & $\\pm{ci:.{d}f}$ & {share:.0f}\\% \\\\")
        L.append("\\midrule" if col == "gap_vs_driver" else "")
    (TAB / "paired.tex").write_text("\n".join(L), encoding="utf-8")


def table_ablation():
    a = pd.read_csv(R / "final_f6_ablation.csv"); base = f"zlns:order=ref:w=3:ii=1000:wi=1500:kmax=2:l=60"
    rows = [("window of 1 zone (no overlap)", "w=1:ii=1000:wi=1500:kmax=2"), ("window of 2 zones", "w=2:ii=1000:wi=1500:kmax=2"), ("window of 3 zones (default)", "w=3:ii=1000:wi=1500:kmax=2"),
            ("window of 4 zones", "w=4:ii=1000:wi=1500:kmax=2"), ("window of 5 zones", "w=5:ii=1000:wi=1500:kmax=2"),
            ("500 instead of 1,500 iterations per window", "w=3:ii=1000:wi=500:kmax=2"), ("4,500 instead of 1,500 iterations per window", "w=3:ii=1000:wi=4500:kmax=2"),
            ("300 instead of 1,000 iterations per zone", "w=3:ii=300:wi=1500:kmax=2"), ("two sweeps over the windows", "w=3:ii=1000:wi=1500:kmax=2:s=2"),
            ("small neighbourhood (up to $m/8$ stops removed)", "w=3:ii=1000:wi=1500")]
    L = []
    for lab, k in rows:
        spec = f"zlns:order=ref:{k}:l=60".replace(":s=2:l=60", ":l=60:s=2")
        if k == "w=3:ii=1000:wi=1500": spec = "zlns:order=ref:w=3:ii=1000:wi=1500:l=60"
        s = a[a.method == spec]; m, ci, _, _ = _pair(a, "gap_vs_driver", spec, base, 100); m2, ci2, _, _ = _pair(a, "official_score", spec, base, 1)
        L.append(f"{lab} & ${100 * s.gap_vs_driver.mean():.2f}$ & ${s.official_score.mean():.4f}$ & ${m:+.2f}\\pm{ci:.2f}$ & ${m2:+.4f}\\pm{ci2:.4f}$ \\\\")
    (TAB / "ablation.tex").write_text("\n".join(L), encoding="utf-8")


def table_literature():
    d = pd.read_csv(R / "final_f3_literature.csv"); g = lambda m: d[d.method == m]
    rnd = d[d.method.str.startswith("random")].groupby("method").official_score.mean().mean()
    drv = g("driver"); stops = int((drv.n - 1).sum()); L = []          # n counts the depot too
    L.append(f"Driver: mean closed-tour time (s) & {drv.closed_time.mean():,.1f} & 12,250.0 \\\\")
    L.append(f"Driver: number of stops & {stops:,} & 163,177 \\\\")
    L.append(f"Random tours: official score (mean of 10) & {rnd:.4f} & 0.91545 \\\\")
    for lab, m, pt, ps in (("Near-optimal closed tours (LNS, 5,000 it.): time (s)", "lns:i=5000:kmax=2:cl=1", "10,853.3 (optimal)", "0.07030"),
                           ("Near-optimal closed tours (LNS, 15,000 it.): time (s)", "lns:i=15000:kmax=2:cl=1", "10,853.3 (optimal)", "0.07030"),
                           ("Zone-clustered closed tours (LNS, $\\lambda=3000$ s, 5,000 it.): time (s)", "lns:i=5000:kmax=2:cl=1:l=3000", "11,235.1 (optimal)", "0.04866"),
                           ("Zone-clustered closed tours (LNS, $\\lambda=30000$ s, 15,000 it.): time (s)", "lns:i=15000:kmax=2:cl=1:l=30000", "11,235.1 (optimal)", "0.04866")):
        x = g(m); L.append(f"{lab} & {x.closed_time.mean():,.1f} & {pt} \\\\"); L.append(f"\\hspace{{1.5em}}official score of these tours & {x.official_score.mean():.4f} & {ps} \\\\")
    for lab, m in (("Zonal LNS, past-routes order ($\\lambda=60$): official score", f"zlns:order=ref:{ZL}:l=60"), ("Plain LNS ($\\lambda=60$): official score", f"{LNS}:l=60")):
        L.append(f"{lab} & {g(m).official_score.mean():.4f} & {'0.03171--0.03322' if 'zonal' in lab else '--'} \\\\")
    (TAB / "literature.tex").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    tab = stats_table(); tab.to_csv(OUT / "main_table.csv", index=False); print(tab.round(4).to_string())
    table_main(tab); table_paired(); table_ablation(); table_literature()
    fig_tradeoff(tab); fig_lambda(); fig_time(); fig_ablation(); fig_pipeline(); fig_data(); fig_routes()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
