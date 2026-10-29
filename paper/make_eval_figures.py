"""Tables and figure for the report from the EVALUATION-folder results (results/eval_*.csv). Run from the repo root:
  python paper/make_eval_figures.py                       # full results: eval_full_s0..s2.csv + eval_full_ortools.csv
  python paper/make_eval_figures.py --pilot               # test of this script on the 300-route pilot (one seed, no OR-Tools)
Writes paper/tables/eval_main.tex, paper/tables/eval_paired.tex and paper/figures/fig_eval_tradeoff.pdf."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results"); TAB = Path("paper/tables"); FIG = Path("paper/figures"); TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
plt.rcParams.update({"text.color": "k", "axes.labelcolor": "k", "xtick.color": "k", "ytick.color": "k", "axes.edgecolor": "k", "font.size": 9, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
LNS = "lns:i=5000:kmax=2"; ZL = "w=3:ii=1000:wi=1500:kmax=2"; ZS = "w=3:ii=50000:wi=300000:t0=0.5"
METHODS = {"plain LNS": (LNS, "#1b6ca8", "o"), "zonal LNS, order by zone routing": (f"zlns:order=ztsp:{ZL}", "#e08214", "^"),
           "zonal LNS, order from past routes": (f"zlns:order=ref:{ZL}", "#b2182b", "D"), "zonal SA, order by zone routing": (f"zsa:order=ztsp:{ZS}", "#fdb863", "v"),
           "zonal SA, order from past routes": (f"zsa:order=ref:{ZS}", "#762a83", "P")}
(FIG / "preview").mkdir(exist_ok=True)
_savefig = matplotlib.figure.Figure.savefig


def _save_with_preview(self, fname, *a, **k):          # every figure is also written as a PNG preview
    _savefig(self, fname, *a, **k); _savefig(self, FIG / "preview" / (Path(fname).stem + ".png"), dpi=130, bbox_inches="tight")


matplotlib.figure.Figure.savefig = _save_with_preview
ORT = {0: "ortools:t=10", 60: "ortools:t=10:l=60"}


def wrap(name, colspec, header, body):
    (TAB / f"{name}.tex").write_text(f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body.strip()}\n\\bottomrule\n\\end{{tabular}}\n", encoding="utf-8")


def pm(m, s, d):
    return f"${m:.{d}f}$" if s is None or np.isnan(s) else f"${m:.{d}f}\\pm{s:.{d}f}$"


def load(a):
    if a.pilot:
        runs = {0: pd.read_csv(R / "eval_pilot_s0.csv")}; ort = None
    else:
        runs = {s: pd.read_csv(R / f"eval_full_s{s}.csv") for s in a.seeds}
        ort = pd.read_csv(R / "eval_full_ortools.csv") if (R / "eval_full_ortools.csv").exists() else None
    return runs, ort


def per_seed(runs, spec):
    rows = []
    for s, d in runs.items():
        x = d[d.method == spec]
        if len(x): rows.append(dict(seed=s, t=100 * x.gap_vs_driver.mean(), z=x.zone_sw.mean(), sc=x.official_score.mean(), n=x.route.nunique()))
    return pd.DataFrame(rows)


def main_rows(runs, ort):
    out = []
    for lam in (0, 60):
        if ort is not None and ORT[lam] in set(ort.method):
            o = ort[ort.method == ORT[lam]]
            out.append(dict(lam=lam, method="OR-Tools (10 s)", t=100 * o.gap_vs_driver.mean(), t_sd=np.nan, z=o.zone_sw.mean(), z_sd=np.nan, sc=o.official_score.mean(), sc_sd=np.nan, seeds=1))
        for lab, (spec, _, _) in METHODS.items():
            p = per_seed(runs, f"{spec}:l={lam}")
            out.append(dict(lam=lam, method=lab, t=p.t.mean(), t_sd=p.t.std(), z=p.z.mean(), z_sd=p.z.std(), sc=p.sc.mean(), sc_sd=p.sc.std(), seeds=len(p)))
    return pd.DataFrame(out)


def table_main(tab):
    L = []
    for lam in (0, 60):
        L.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{'$\\lambda=0$ (travel time only)' if lam == 0 else '$\\lambda=60$ s (balanced)'}}}}}\\\\")
        for _, r in tab[tab.lam == lam].iterrows():
            L.append(f"{r.method} & {pm(r.t, r.t_sd, 2)} & {pm(r.z, r.z_sd, 1)} & {pm(r.sc, r.sc_sd, 4)} \\\\")
        if lam == 0: L.append("\\midrule")
    wrap("eval_main", "lR{3.1cm}R{2.4cm}R{2.9cm}", "Method & Travel time vs.\\ driver (\\%) & Zone changes & Official score", "\n".join(L))


def _pair(df, col, a, b, scale):
    w = df.pivot(index="route", columns="method", values=col) * scale; d = (w[a] - w[b]).dropna()
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d)), 100 * (d < 0).mean()


def table_paired(runs, ort):
    df = runs[min(runs)]; df = pd.concat([df, ort]).drop_duplicates(["route", "method"]) if ort is not None else df
    ZLx = lambda o, l: f"zlns:order={o}:{ZL}:l={l}"; ZSx = lambda o, l: f"zsa:order={o}:{ZS}:l={l}"; have = set(df.method)
    t_rows = [("$\\lambda=0$: plain LNS vs OR-Tools", f"{LNS}:l=0", ORT[0]), ("$\\lambda=0$: zonal LNS (zone-routing order) vs OR-Tools", ZLx("ztsp", 0), ORT[0]),
              ("$\\lambda=0$: zonal LNS (zone-routing order) vs plain LNS", ZLx("ztsp", 0), f"{LNS}:l=0"), ("$\\lambda=0$: zonal SA vs zonal LNS (zone-routing order)", ZSx("ztsp", 0), ZLx("ztsp", 0)),
              ("$\\lambda=60$: zonal SA vs zonal LNS (past-routes order)", ZSx("ref", 60), ZLx("ref", 60)), ("$\\lambda=60$: plain LNS vs OR-Tools", f"{LNS}:l=60", ORT[60]),
              ("$\\lambda=60$: zonal LNS (past-routes order) vs OR-Tools", ZLx("ref", 60), ORT[60]), ("$\\lambda=60$: zonal LNS (past-routes order) vs plain LNS", ZLx("ref", 60), f"{LNS}:l=60")]
    s_rows = [("$\\lambda=60$: zonal LNS (past-routes order) vs OR-Tools", ZLx("ref", 60), ORT[60]), ("$\\lambda=60$: zonal LNS (past-routes order) vs plain LNS", ZLx("ref", 60), f"{LNS}:l=60"),
              ("$\\lambda=60$: zonal LNS, past-routes vs zone-routing order", ZLx("ref", 60), ZLx("ztsp", 60))]
    L = ["\\multicolumn{4}{l}{\\textit{travel time (points)}}\\\\"]
    for lab, a, b in t_rows:
        if a in have and b in have:
            m, ci, sh = _pair(df, "gap_vs_driver", a, b, 100); L.append(f"{lab} & ${m:+.2f}$ & $\\pm{ci:.2f}$ & {sh:.0f}\\% \\\\")
    L += ["\\midrule", "\\multicolumn{4}{l}{\\textit{official score}}\\\\"]
    for lab, a, b in s_rows:
        if a in have and b in have:
            m, ci, sh = _pair(df, "official_score", a, b, 1); L.append(f"{lab} & ${m:+.4f}$ & $\\pm{ci:.4f}$ & {sh:.0f}\\% \\\\")
    wrap("eval_paired", "p{8.6cm}R{2.0cm}R{2.2cm}R{2.0cm}", "Comparison & Difference & 95\\% interval & First better", "\n".join(L))


def fig_tradeoff(tab):
    fig, ax = plt.subplots(figsize=(5.8, 4.3)); ax.scatter([0], [0], marker="*", s=140, c="k", zorder=5, label="driver (reference)")
    cols = {"OR-Tools (10 s)": ("#2ca02c", "X")}; cols.update({k: (v[1], v[2]) for k, v in METHODS.items()})
    for lab, (c, mk) in cols.items():
        d = tab[tab.method == lab].sort_values("lam")
        if d.empty: continue
        ax.plot(d.t, d.sc, "-", c=c, lw=1, alpha=.7)
        ax.scatter(d.t[d.lam == 0], d.sc[d.lam == 0], marker=mk, s=45, facecolors="none", edgecolors=c, linewidths=1.5, zorder=4)
        ax.scatter(d.t[d.lam == 60], d.sc[d.lam == 60], marker=mk, s=45, c=c, zorder=4, label=lab)
    ax.set_xlabel("travel time relative to the driver (%)   [left = faster]"); ax.set_ylabel("official score   [lower = more like the driver]")
    ax.text(0.01, 0.99, "hollow marker: $\\lambda=0$, filled marker: $\\lambda=60$ s", transform=ax.transAxes, va="top", fontsize=8)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, frameon=False); ax.invert_xaxis(); fig.tight_layout(); fig.savefig(FIG / "fig_eval_tradeoff.pdf"); plt.close(fig)


def route_means(runs, ort):
    """One row per (route, method): mean over seeds of travel time vs driver, official score and zone changes."""
    d = pd.concat(list(runs.values()))
    g = d.groupby(["route", "method"]).agg(gap=("gap_vs_driver", "mean"), sc=("official_score", "mean"), zsw=("zone_sw", "mean"), n=("n", "first"), station=("station", "first")).reset_index()
    if ort is not None:
        o = ort.groupby(["route", "method"]).agg(gap=("gap_vs_driver", "mean"), sc=("official_score", "mean"), zsw=("zone_sw", "mean"), n=("n", "first"), station=("station", "first")).reset_index()
        g = pd.concat([g, o])
    return g


TRIO = [("OR-Tools (10 s)", ORT[60], "#2ca02c"), ("plain LNS", f"{LNS}:l=60", "#1b6ca8"), ("zonal LNS, order from past routes", f"zlns:order=ref:{ZL}:l=60", "#b2182b")]


def fig_eval_depot(g):
    """Results of three methods (lambda=60 s) for every depot: travel time and official score."""
    cnt = g[g.method == "driver"].groupby("station").route.nunique().sort_values(ascending=False); cnt = cnt[cnt >= 10]; deps = list(cnt.index)
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 4.3), sharey=True); y = np.arange(len(deps)); h = 0.26
    for k, (lab, spec, c) in enumerate(TRIO):
        x = g[g.method == spec].groupby("station").agg(t=("gap", lambda v: 100 * v.mean()), s=("sc", "mean")).reindex(deps)
        ax[0].barh(y + (k - 1) * h, x.t, h, color=c, label=lab); ax[1].barh(y + (k - 1) * h, x.s, h, color=c)
    ax[0].set_yticks(y); ax[0].set_yticklabels([f"{d} ({cnt[d]})" for d in deps], fontsize=7); ax[0].invert_yaxis()
    ax[0].set_xlabel("travel time relative to the driver (%)"); ax[1].set_xlabel("official score (lower = more like the driver)")
    ax[0].legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0.0, -0.30), ncol=3, frameon=False, columnspacing=1.0, handlelength=1.2)
    for a in ax: a.grid(axis="y", alpha=0)
    fig.tight_layout(); fig.savefig(FIG / "fig_eval_depot.pdf", bbox_inches="tight"); plt.close(fig)


def fig_eval_dist(g):
    """Spread over routes: histogram of the official score and travel-time saving by route size (lambda=60 s)."""
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6)); w = g[g.method == "driver"].set_index("route").n
    for lab, spec, c in TRIO:
        x = g[g.method == spec].set_index("route"); ax[0].hist(x.sc.clip(upper=0.25), bins=np.linspace(0, 0.25, 31), histtype="step", color=c, lw=1.4, label=lab)
        bins = pd.qcut(w.loc[x.index] - 1, 6); m = x.groupby(bins, observed=True).agg(t=("gap", lambda v: 100 * v.mean())); ctr = [(iv.left + iv.right) / 2 for iv in m.index]
        ax[1].plot(ctr, m.t, "o-", c=c, ms=4, lw=1.2)
    ax[0].set_xlabel("official score of a route (clipped at 0.25)"); ax[0].set_ylabel("routes"); ax[1].set_xlabel("stops in the route (6 equal-sized groups)"); ax[1].set_ylabel("travel time vs driver (%)")
    ax[0].legend(fontsize=6.5, frameon=False); fig.tight_layout(); fig.savefig(FIG / "fig_eval_dist.pdf"); plt.close(fig)


def fig_dev_studies():
    """Studies on the 207 development routes (final_f*.csv): effect of lambda, window width, run time."""
    f1 = pd.read_csv(R / "final_f1_test.csv"); ab = pd.read_csv(R / "final_f6_ablation.csv"); tm = pd.read_csv(R / "final_f4_timing.csv")
    fig = plt.figure(figsize=(7.0, 4.6)); gs = fig.add_gridspec(2, 6); axl = [fig.add_subplot(gs[0, 2 * i:2 * i + 2]) for i in range(3)]
    lams = [0, 15, 30, 60, 120]
    g = [100 * f1[f1.method == f"{LNS}:l={l}"].gap_vs_driver.mean() for l in lams]; z = [f1[f1.method == f"{LNS}:l={l}"].zone_sw.mean() for l in lams]
    s = [f1[f1.method == f"{LNS}:l={l}"].official_score.mean() for l in lams]; zd = f1[f1.method == "driver"].zone_sw.mean()
    for a, v, t in zip(axl, (g, z, s), ("(a) travel time vs driver (%)", "(b) zone changes per route", "(c) official score")):
        a.plot(lams, v, "o-", c="#1b6ca8", ms=4); a.set_xlabel("$\\lambda$ (s per zone change)", fontsize=8); a.set_title(t, fontsize=8)
    axl[0].axhline(0, c="k", lw=.8); axl[1].axhline(zd, c="k", ls="--", lw=.8); axl[1].text(lams[-1], zd + .8, "driver", ha="right", fontsize=7)
    a1, a2, a3 = fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:6])
    for order, lab, c in (("ref", "order from past routes", "#b2182b"), ("ztsp", "order by zone routing", "#e08214")):
        ws = [1, 2, 3, 4, 5]; sub = [ab[ab.method == f"zlns:order={order}:w={w}:ii=1000:wi=1500:kmax=2:l=60"] for w in ws]
        a1.plot(ws, [100 * x.gap_vs_driver.mean() for x in sub], "o-", c=c, ms=4, label=lab); a2.plot(ws, [x.official_score.mean() for x in sub], "o-", c=c, ms=4)
    a1.set_title("(d) travel time vs driver (%)", fontsize=8); a2.set_title("(e) official score", fontsize=8); a1.legend(fontsize=6, frameon=False)
    for a in (a1, a2): a.set_xlabel("window width $w$ (zones)", fontsize=8); a.set_xticks([1, 2, 3, 4, 5])
    rows = [("OR-Tools 10 s", "ortools:t=10"), ("OR-Tools 2 s", "ortools:t=2"), ("plain LNS", LNS), ("zonal SA", f"zsa:order=ztsp:{ZS}"), ("zonal LNS", f"zlns:order=ztsp:{ZL}")]
    v = [tm[tm.method == m].secs.mean() for _, m in rows]; yy = np.arange(len(rows))
    a3.barh(yy, v, color="#1b6ca8"); a3.set_xscale("log"); a3.set_yticks(yy); a3.set_yticklabels([l for l, _ in rows], fontsize=7); a3.invert_yaxis(); a3.grid(axis="y", alpha=0)
    for yi, vi in zip(yy, v): a3.text(vi * 1.1, yi, f"{vi:.2f}", va="center", fontsize=7)
    a3.set_xlim(0.05, 60); a3.set_title("(f) seconds per route (log)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig_dev_studies.pdf"); plt.close(fig)


def fig_design():
    """Which routes are used for what (replaces a table)."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(6.8, 2.9)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.grid(False)
    def box(x, y, w, h, txt, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", fc=fc, ec="#444444", lw=.9)); ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7.2)
    def arrow(x0, y0, x1, y1): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=.9, color="#333333"))
    box(1, 15, 20, 14, "Training folder\n6,112 routes with\nthe drivers' sequences", "#e8e8e8")
    box(33, 34, 30, 9, "60 validation routes\nsettings were tuned here", "#dbe9f6")
    box(33, 21.5, 30, 10, "207 development routes\n$\\lambda$ study, ablation, run time\n(5 seeds)", "#dbe9f6")
    box(33, 8.5, 30, 10.5, "library of past routes\n3,691 routes (development)\n6,112 routes (evaluation)", "#fde3c8")
    box(1, 1.5, 20, 8, "Evaluation folder\n3,052 routes", "#e8e8e8")
    box(76, 5, 23, 24, "Final results\n3,052 routes\n3 seeds of our methods,\nOR-Tools once", "#d9efd5")
    for y in (38.5, 26.5, 13.5): arrow(21.5, 22, 32.3, y)
    arrow(63.5, 13.5, 75.3, 17); arrow(21.5, 5.5, 75.3, 8.5)
    ax.text(48, 0.0, "no setting was chosen using evaluation routes", fontsize=6.8, style="italic", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(FIG / "fig_design.pdf"); plt.close(fig)


def eval_numbers():
    from lmr.data import load_eval_routes
    rs = load_eval_routes(); n = np.array([len(r.T) - 1 for r in rs]); z = np.array([len(set(r.zone[1:].tolist())) for r in rs])
    info = dict(routes=len(rs), stations=len({r.station for r in rs}), stops_total=int(n.sum()), stops_mean=n.mean(), stops_min=n.min(), stops_max=n.max(), zones_mean=z.mean(), zones_min=z.min(), zones_max=z.max())
    pd.Series(info).to_csv(FIG / "eval_numbers.csv", header=False); print(pd.Series(info).round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pilot", action="store_true"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); a = ap.parse_args()
    runs, ort = load(a); tab = main_rows(runs, ort); tab.to_csv(R / ("eval_pilot_table.csv" if a.pilot else "eval_main_table.csv"), index=False)
    d0 = runs[min(runs)]; drv = d0[d0.method == "driver"]
    print(f"routes {drv.route.nunique()}, driver zone changes {drv.zone_sw.mean():.2f}, seeds {sorted(runs)}, OR-Tools {'yes' if ort is not None else 'no'}")
    print(tab.round(4).to_string()); table_main(tab); table_paired(runs, ort); fig_tradeoff(tab)
    if not a.pilot:
        g = route_means(runs, ort); fig_eval_depot(g); fig_eval_dist(g); fig_dev_studies(); fig_design(); eval_numbers()
