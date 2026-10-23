"""Tables and statistics for the report. All comparisons are paired by route."""
import numpy as np, pandas as pd

SUMMARY_COLS = ["travel_time", "gap_vs_driver", "zone_sw", "zone_blocks_per_zone", "sim_to_driver",
                "zone_order_sim", "duration", "late_stops", "secs"]


def summary(df, cols=SUMMARY_COLS):
    """Mean per method. gap_vs_driver is in fractions (-0.10 = 10% less travel time than the driver)."""
    cols = [c for c in cols if c in df.columns]
    out = df.groupby("method")[cols].mean()
    out.insert(0, "n_routes", df.groupby("method").route.nunique())
    return out.sort_values("gap_vs_driver")


def wide(df, col="travel_time"):
    return df.pivot(index="route", columns="method", values=col)


def paired(df, a, b, col="travel_time", relative_to="driver_time"):
    """Method a minus method b on the same routes, in percentage points of the driver's value
    (so -2.0 means a is 2 points of driver travel time better). Returns mean, 95% CI half-width (normal
    approximation, 1.96 * standard error), the share of routes where a is strictly better, and n."""
    w = wide(df, col); base = df.drop_duplicates("route").set_index("route")[relative_to]
    d = ((w[a] - w[b]) / base * 100).dropna()
    return dict(a=a, b=b, mean_pts=d.mean(), ci95=1.96 * d.sem(), a_better=float((d < -1e-9).mean()),
                tie=float((d.abs() <= 1e-9).mean()), n=len(d))


def paired_table(df, pairs, **kw):
    return pd.DataFrame([paired(df, a, b, **kw) for a, b in pairs]).round(3)


def penalised(df, lam):
    """Travel time plus lam seconds per zone change (the objective the penalised methods minimise)."""
    return df.assign(objective=df.travel_time + lam * df.zone_sw)


def tradeoff(df, methods=None):
    """Mean travel-time gap vs driver and mean zone changes per method: the points of the trade-off curve."""
    s = summary(df)[["gap_vs_driver", "zone_sw", "zone_blocks_per_zone", "secs"]]
    return s if methods is None else s.loc[methods]
