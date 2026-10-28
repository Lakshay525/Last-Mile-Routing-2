"""Generate every results table of docs/RESULTS.md from the CSV files (no number is typed by hand).
Run from the repo root:  python scripts/make_tables.py > docs/RESULTS_tables.md
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from lmr.analysis import summary, paired

R = Path("results")
LNS, SA = "lns:i=5000:kmax=2", "sa:i=5000000:t0=0.5"
ZL, ZS = ":w=3:ii=1000:wi=1500:kmax=2", ":w=3:ii=50000:wi=300000:t0=0.5"
G20 = "gnn:beam=20"


def load(*names):
    return pd.concat([pd.read_csv(R / f"{n}.csv") for n in names if (R / f"{n}.csv").exists()]).drop_duplicates(["route", "method"])


def md(df, floatfmt=None):
    cols = list(df.columns); out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join((f"{v:.{floatfmt[c]}f}" if floatfmt and c in floatfmt and isinstance(v, (int, float, np.floating)) and not pd.isna(v) else ("" if (isinstance(v, float) and pd.isna(v)) else str(v))) for c, v in r.items()) + " |")
    return "\n".join(out)


def table(df, rows, extra=None):
    s = summary(df, ["gap_vs_driver", "zone_sw", "sim_to_driver", "official_score", "secs"])
    t = pd.DataFrame({"method": [lab for lab, m in rows if m in s.index],
                      "travel time vs driver %": [100 * s.loc[m, "gap_vs_driver"] for lab, m in rows if m in s.index],
                      "zone changes": [s.loc[m, "zone_sw"] for lab, m in rows if m in s.index],
                      "similarity to driver": [s.loc[m, "sim_to_driver"] for lab, m in rows if m in s.index],
                      "official score (lower=better)": [s.loc[m, "official_score"] for lab, m in rows if m in s.index]})
    return md(t, {"travel time vs driver %": 2, "zone changes": 1, "similarity to driver": 3, "official score (lower=better)": 4})


def main():
    test = load("final_f1_test", "final_f5_gnn_order", "final_f2_ortools")
    old_or = pd.read_csv(R / "exp13_ortools.csv"); old_or["method"] = "[run 1] " + old_or.method
    print(f"## Test routes: {test.route.nunique()} routes (hash split), seed 0\n")

    print("### T1. Travel time first (lambda = 0)\n")
    print(table(test, [("driver", "driver"), ("nearest neighbour", "nn"), ("OR-Tools 2 s", "ortools:t=2"), ("OR-Tools 10 s", "ortools:t=10"),
                       ("**plain LNS (approach 2)**", f"{LNS}:l=0"), ("global SA", f"{SA}:l=0"),
                       ("zonal LNS, zone-level routing order", f"zlns:order=ztsp{ZL}:l=0"), ("zonal LNS, reference-route order", f"zlns:order=ref{ZL}:l=0"),
                       ("zonal LNS, GNN+reference order", f"zlns:order={G20}{ZL}:l=0"), ("zonal SA (third approach), zone-level routing order", f"zsa:order=ztsp{ZS}:l=0"),
                       ("zonal SA, reference-route order", f"zsa:order=ref{ZS}:l=0"), ("zonal LNS, driver's order (oracle, not deployable)", f"zlns:order=driver{ZL}:l=0")]))
    o1 = old_or[old_or.method == "[run 1] ortools:t=10"]
    print(f"\nOR-Tools 10 s was run twice (time-limited, so results vary): run 1 {100*o1.gap_vs_driver.mean():.2f}% travel time vs driver; "
          f"run 2 (table above) {100*test[test.method=='ortools:t=10'].gap_vs_driver.mean():.2f}%.\n")

    print("### T2. Balanced objective (zone-change penalty lambda = 60 s)\n")
    print(table(test, [("driver", "driver"), ("OR-Tools 10 s", "ortools:t=10:l=60"), ("plain LNS", f"{LNS}:l=60"), ("global SA", f"{SA}:l=60"),
                       ("zonal LNS, zone-level routing order", f"zlns:order=ztsp{ZL}:l=60"), ("zonal LNS, greedy order", f"zlns:order=greedy{ZL}:l=60"),
                       ("zonal LNS, reference-route order", f"zlns:order=ref{ZL}:l=60"), ("zonal LNS, GNN+reference order", f"zlns:order={G20}{ZL}:l=60"),
                       ("zonal SA, reference-route order", f"zsa:order=ref{ZS}:l=60"), ("zonal SA, GNN+reference order", f"zsa:order={G20}{ZS}:l=60"),
                       ("zonal LNS, driver's order (oracle)", f"zlns:order=driver{ZL}:l=60")]))

    print("\n### T3. Zone-change penalty sweep, plain LNS\n")
    print(table(test, [("driver", "driver")] + [(f"lambda = {l}", f"{LNS}:l={l}") for l in (0, 15, 30, 60, 120)]))

    print("\n### T4. Soft zone-order penalty rho (reference-route order, then global LNS refinement of 5,000 iterations)\n")
    print(table(test, [(f"rho = {r}", f"zlns:order=ref{ZL}:g=5000:rho={r}") for r in (30, 100, 300)] +
               [("no refinement (windows only), lambda=0", f"zlns:order=ref{ZL}:l=0"), ("GNN+ref, rho = 100", f"zlns:order={G20}{ZL}:g=5000:rho=100")]))


    print("\n### T5. Paired differences (percentage points of the driver's travel time; negative = first method has LESS travel time)\n")
    P = [(f"{LNS}:l=0", "ortools:t=10", "plain LNS vs OR-Tools 10 s"), (f"{SA}:l=0", "ortools:t=10", "global SA vs OR-Tools 10 s"),
         (f"{SA}:l=0", f"{LNS}:l=0", "global SA vs plain LNS"), (f"zlns:order=ztsp{ZL}:l=0", "ortools:t=10", "zonal LNS (ztsp) vs OR-Tools 10 s"),
         (f"zsa:order=ztsp{ZS}:l=0", f"zlns:order=ztsp{ZL}:l=0", "zonal SA vs zonal LNS (same order)"),
         (f"zlns:order=ztsp{ZL}:l=0", f"{LNS}:l=0", "zonal LNS (ztsp) vs plain LNS"), (f"{LNS}:l=60", "ortools:t=10:l=60", "plain LNS vs OR-Tools, lambda=60"),
         (f"zlns:order=ref{ZL}:l=60", f"{LNS}:l=60", "zonal LNS (ref) vs plain LNS, lambda=60"), (f"zlns:order=ref{ZL}:l=60", "ortools:t=10:l=60", "zonal LNS (ref) vs OR-Tools, lambda=60"),
         (f"zlns:order={G20}{ZL}:l=60", f"zlns:order=ref{ZL}:l=60", "zonal LNS: GNN+ref order vs ref order, lambda=60")]
    rows = []
    for a, b, lab in P:
        if a in set(test.method) and b in set(test.method):
            x = paired(test, a, b); rows.append(dict(comparison=lab, **{"mean difference (points)": x["mean_pts"], "95% interval +-": x["ci95"], "first better on % of routes": 100 * x["a_better"], "n": x["n"]}))
    print(md(pd.DataFrame(rows), {"mean difference (points)": 2, "95% interval +-": 2, "first better on % of routes": 0}))
    print("\nOfficial score, paired (negative = first method scores better):\n")
    w = test.pivot(index="route", columns="method", values="official_score"); rows = []
    for a, b, lab in [(f"zlns:order=ref{ZL}:l=60", f"{LNS}:l=60", "zonal LNS (ref) vs plain LNS, lambda=60"), (f"zlns:order=ref{ZL}:l=60", "ortools:t=10:l=60", "zonal LNS (ref) vs OR-Tools, lambda=60"),
                      (f"zlns:order={G20}{ZL}:l=60", f"zlns:order=ref{ZL}:l=60", "GNN+ref vs ref order, lambda=60"), (f"zlns:order=ref{ZL}:l=60", f"zlns:order=ztsp{ZL}:l=60", "ref order vs zone-level routing order")]:
        if a in w and b in w:
            d = (w[a] - w[b]).dropna(); rows.append(dict(comparison=lab, **{"mean difference": d.mean(), "95% interval +-": 1.96 * d.sem(), "first better on % of routes": 100 * (d < 0).mean(), "n": len(d)}))
    print(md(pd.DataFrame(rows), {"mean difference": 4, "95% interval +-": 4, "first better on % of routes": 0}))

    zj = R / "zone_gnn_ref.json"
    if zj.exists():
        print("\n### T6. Zone-order prediction (Kendall tau between predicted and the driver's zone order, 207 test routes)\n")
        j = json.load(open(zj)); rows = [dict(order=("GNN WITH reference-route features: " + k) if k.startswith("gnn") else k, mean=v["mean_tau"], median=v["median_tau"], sd=v["std"]) for k, v in j["test"].items()]
        gj = R / "zone_gnn_geom.json"
        if gj.exists():
            jg = json.load(open(gj)); rows += [dict(order="GNN geometry only: " + k, mean=v["mean_tau"], median=v["median_tau"], sd=v["std"]) for k, v in jg["test"].items() if k.startswith("gnn")]
        print(md(pd.DataFrame(rows), {"mean": 3, "median": 3, "sd": 3}))

    lit = R / "final_f3_literature.csv"
    if lit.exists():
        df = pd.read_csv(lit); d = df[df.method == "driver"]
        print(f"\n## Comparison with the literature: {df.route.nunique()} High+Delivered routes (paper: 1,107)\n")
        rnd = df[df.method.str.startswith("random")].groupby("method").official_score.mean().mean()
        g = df[~df.method.str.startswith("random")].groupby("method").agg(closed=("closed_time", "mean"), score=("official_score", "mean"), gap=("gap_vs_driver", lambda x: 100 * x.mean()), zsw=("zone_sw", "mean"), sim=("sim_to_driver", "mean"))
        pap = {"driver": ("12,250.0", "0"), "lns:i=5000:kmax=2:cl=1": ("10,853.3 (optimal)", "0.07030"), "lns:i=5000:kmax=2:cl=1:l=3000": ("11,235.1 (optimal, zone-clustered)", "0.04866")}
        names = {"lns:i=5000:kmax=2:cl=1": "closed LNS, no zone penalty (near-optimal ATSP tours)", "lns:i=15000:kmax=2:cl=1": "closed LNS, 15,000 iterations",
                 "lns:i=5000:kmax=2:cl=1:l=3000": "closed LNS, lambda=3000 (zone-clustered)", "lns:i=15000:kmax=2:cl=1:l=30000": "closed LNS, lambda=30000, 15,000 iterations (zone-clustered)",
                 f"{LNS}:l=60": "plain LNS lambda=60 (open path)", f"zlns:order=ref{ZL}:l=60": "zonal LNS, reference-route order, lambda=60", f"zlns:order=driver{ZL}:l=60": "zonal LNS, driver's order (oracle), lambda=60",
                 f"zlns:order=ztsp{ZL}:l=60": "zonal LNS, zone-level routing order, lambda=60", f"{SA}:l=60": "global SA lambda=60", "driver": "driver", "nn": "nearest neighbour"}
        t = pd.DataFrame([dict(method=names[m], **{"mean closed-tour time (s)": g.loc[m, "closed"], "official score": g.loc[m, "score"], "zone changes": g.loc[m, "zsw"]}) for m in names if m in g.index])
        print(md(t, {"mean closed-tour time (s)": 1, "official score": 5, "zone changes": 1}))
        print(f"\nRandom tours, official score, mean of 10 trials: {rnd:.5f} (paper 0.91545). Driver stops: {int(d.n.sum() - len(d))} (paper 163,177).")
        print("Paper (Cook et al.): optimal ATSP tours 10,853.3 s / 0.07030; optimal zone-clustered tours 11,235.1 s / 0.04866; driver zone order with LKH-AMZ 0.00510-0.00738; "
              "reference-route precedence constraints 0.03171-0.03322.")

    tm = R / "final_f4_timing.csv"
    if tm.exists():
        d = pd.read_csv(tm); t = d.groupby("method").agg(mean_secs=("secs", "mean"), median_secs=("secs", "median"), gap=("gap_vs_driver", lambda x: 100 * x.mean()), score=("official_score", "mean"), routes=("route", "nunique")).reset_index()
        print("\n### T7. Clean timings (20 test routes, one process, nothing else running)\n"); print(md(t, {"mean_secs": 2, "median_secs": 2, "gap": 2, "score": 4}))

    sm, sf = R / "exp40_solver_study_small.csv", R / "exp40_solver_study_full.csv"
    if sm.exists():
        d = pd.read_csv(sm); d["best"] = d[["lns_cost", "cpsat_cost", "mip_cost"]].min(axis=1)
        t = d.groupby("k").apply(lambda x: pd.Series({"routes": len(x), "CP-SAT proved optimal": int((x.cpsat_status == "OPTIMAL").sum()), "MIP proved optimal": int((x.mip_status == "OPTIMAL").sum()),
                                                      "CP-SAT mean s": x.cpsat_secs.mean(), "MIP mean s": x.mip_secs.mean(), "LNS found best known %": 100 * (x.lns_cost <= x.best + 1e-6).mean(),
                                                      "CP-SAT found best known %": 100 * (x.cpsat_cost <= x.best + 1e-6).mean(), "MIP found best known %": 100 * (x.mip_cost.fillna(1e18) <= x.best + 1e-6).mean(),
                                                      "LNS excess over best %": 100 * (x.lns_cost / x.best - 1).mean()}), include_groups=False).reset_index()
        print("\n### T8. OR-Tools solvers on small instances (first k stops of validation routes, 20 s limit, 1 worker each)\n"); print(md(t, {c: (0 if c in ("k", "routes", "CP-SAT proved optimal", "MIP proved optimal") else 1) for c in t.columns}))
    if sf.exists():
        f = pd.read_csv(sf); f["imp"] = 100 * (1 - f.cpsat_cost / f.lns_cost); f["gap"] = 100 * (f.cpsat_cost / f.cpsat_bound - 1)
        print(f"\nFull-size routes ({len(f)} validation routes, CP-SAT 60 s, 2 workers, warm start from LNS): CP-SAT improves the LNS route by {f.imp.mean():.2f}% on average "
              f"(median {f.imp.median():.2f}%, max {f.imp.max():.2f}%); proved optimal on {(f.cpsat_status=='OPTIMAL').sum()} of {len(f)}; median proven optimality gap {f.gap.median():.1f}%.")

    ab = R / "final_f6_ablation.csv"
    if ab.exists():
        a = pd.read_csv(ab); sa_ = summary(a, ["gap_vs_driver", "zone_sw", "official_score", "secs"]); sc = a.pivot(index="route", columns="method", values="official_score")
        B = f"zlns:order=ref{ZL}:l=60"; rows = []
        def vr(m, lab):
            x = paired(a, m, B) if m != B else dict(mean_pts=0.0, ci95=0.0); d = (sc[m] - sc[B]).dropna()
            rows.append({"variant": lab, "travel time vs driver %": 100 * sa_.loc[m, "gap_vs_driver"], "zone changes": sa_.loc[m, "zone_sw"], "official score": sa_.loc[m, "official_score"],
                         "travel time vs default (points)": f"{x['mean_pts']:+.2f} +- {x['ci95']:.2f}", "score vs default": f"{d.mean():+.4f} +- {1.96*d.sem():.4f}"})
        for o, on in (("ref", "reference-route order"), ("ztsp", "zone-level routing order")):
            for w in (1, 2, 3, 4, 5):
                vr(f"zlns:order={o}:w={w}:ii=1000:wi=1500:kmax=2:l=60", f"{on}, window of {w} zone(s)" + (" (no overlap)" if w == 1 else ""))
        for m, lab in [("zlns:order=ref:w=3:ii=1000:wi=500:kmax=2:l=60", "reference order, w=3, 500 iterations per window"), ("zlns:order=ref:w=3:ii=1000:wi=4500:kmax=2:l=60", "reference order, w=3, 4,500 iterations per window"),
                       ("zlns:order=ref:w=3:ii=300:wi=1500:kmax=2:l=60", "reference order, w=3, 300 iterations per zone"), ("zlns:order=ref:w=3:ii=1000:wi=1500:kmax=2:l=60:s=2", "reference order, w=3, two sweeps"),
                       ("zlns:order=ref:w=3:ii=1000:wi=1500:l=60", "reference order, w=3, small neighbourhood (kmax=8)"), ("zlns:order=ref:w=3:ii=1000:wi=1500:kmax=2:l=60:g=5000", "reference order, w=3, plus a global LNS refinement (5,000 iterations)")]:
            if m in sa_.index: vr(m, lab)
        print("\n### T9. Ablation of the zonal method (test routes, lambda = 60; the default is the reference-route order with 3-zone windows)\n")
        print(md(pd.DataFrame(rows), {"travel time vs driver %": 2, "zone changes": 1, "official score": 4}))

    hi = R / "final_f7_hierarchy.csv"
    if hi.exists():
        h = pd.read_csv(hi); sh = summary(h, ["gap_vs_driver", "zone_sw", "sim_to_driver", "official_score"]); sc = h.pivot(index="route", columns="method", values="official_score"); rows = []
        Lp, Zp = f"{LNS}:l=60", f"zlns:order=ref{ZL}:l=60"
        for m in h.method.unique():
            if m == "driver": continue
            base = Lp if m.startswith("lns") else Zp; x = paired(h, m, base) if m != base else dict(mean_pts=0.0, ci95=0.0); d = (sc[m] - sc[base]).dropna()
            rows.append({"method": m.replace(LNS, "plain LNS").replace(f"zlns:order=ref{ZL}", "zonal LNS, reference order"), "travel time vs driver %": 100 * sh.loc[m, "gap_vs_driver"], "zone changes": sh.loc[m, "zone_sw"],
                         "official score": sh.loc[m, "official_score"], "travel time vs the lambda=60 row without hl (points)": f"{x['mean_pts']:+.2f} +- {x['ci95']:.2f}", "score vs the lambda=60 row without hl": f"{d.mean():+.4f} +- {1.96*d.sem():.4f}"})
        print("\n### T10. Extra penalty for leaving coarser clusters of the zone ID (hl = seconds per level; lambda = 60 unless shown; test routes)\n")
        print(md(pd.DataFrame(rows), {"travel time vs driver %": 2, "zone changes": 1, "official score": 4}))


if __name__ == "__main__":
    main()
