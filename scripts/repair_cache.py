"""One-off: apply fix_missing_zones to routes already cached in .cache/*.pkl (avoids re-reading the 1.8 GB file)."""
import pickle, sys
from lmr import config
from lmr.data import fix_missing_zones

if __name__ == "__main__":
    for f in sorted(config.CACHE_DIR.glob("routes_*.pkl")):
        rs = pickle.load(open(f, "rb")); n = 0
        for r in rs:
            z = fix_missing_zones(r.zone, r.xy)
            if (z != r.zone).any(): n += 1; r.zone = z
        pickle.dump(rs, open(f, "wb")); print(f.name, "routes repaired:", n, "of", len(rs))
