"""Mean and spread over random seeds for the main methods (seed 0 = final_f1_test.csv, seeds 1-4 = final_f8_seed*.csv).
For each method, the 207 test routes are averaged per seed; the table gives the mean over seeds and the standard deviation
of the per-seed means (how much a re-run with a different seed moves the headline number)."""
import glob
import pandas as pd

LNS = "lns:i=5000:kmax=2"; ZL = "w=3:ii=1000:wi=1500:kmax=2"; ZS = "w=3:ii=50000:wi=300000:t0=0.5"; SA = "sa:i=5000000:t0=0.5"
NAMES = {LNS: "plain LNS", SA: "plain SA", f"zlns:order=ztsp:{ZL}": "zonal LNS (zone order: route-through-zones)",
         f"zlns:order=ref:{ZL}": "zonal LNS (zone order: past routes)", f"zsa:order=ztsp:{ZS}": "zonal SA (zone order: route-through-zones)",
         f"zsa:order=ref:{ZS}": "zonal SA (zone order: past routes)"}

def main(out="results/seed_study.csv"):
    parts = [pd.read_csv("results/final_f1_test.csv").assign(seed=0)] + [pd.read_csv(f).assign(seed=s) for s, f in
             sorted((int(f.split("seed")[-1][:-4]), f) for f in glob.glob("results/final_f8_seed*.csv"))]
    df = pd.concat(parts); rows = []
    for lam in (0, 60):
        for base, name in NAMES.items():
            m = f"{base}:l={lam}"; d = df[df.method == m]
            if d.empty: continue
            per = d.groupby("seed").agg(t=("gap_vs_driver", lambda x: 100 * x.mean()), z=("zone_sw", "mean"), s=("official_score", "mean"))
            rows.append(dict(lam=lam, method=name, seeds=len(per),
                             time_pct=f"{per.t.mean():.2f} +- {per.t.std():.2f}", zone_changes=f"{per.z.mean():.2f} +- {per.z.std():.2f}",
                             score=f"{per.s.mean():.4f} +- {per.s.std():.4f}"))
    t = pd.DataFrame(rows); t.to_csv(out, index=False); print(t.to_string(index=False))

if __name__ == "__main__":
    main()
