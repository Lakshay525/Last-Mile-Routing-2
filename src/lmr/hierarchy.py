"""Hierarchical clusters from the structure of Amazon's zone IDs (Cook, Held & Helsgaun 2022, section 4.6).

A zone ID has the form  G-x.yD  (e.g. "A-2.1E"): a capital letter G, an integer x, an integer y and a capital letter D.
Drivers tend to finish "super-clusters" of zones that agree on some of these four components. A chain of clusterings
S (3 components) > T (2 of them) > U (1 of them) is chosen from the TRAINING routes as the one the drivers cross the
least, and leaving a super-cluster is then penalised on top of leaving a zone.
"""
import itertools, json, re
import numpy as np
from . import config

_PAT = re.compile(r"^([A-Z])-(\d+)\.(\d+)([A-Z])$")


def parse(name):
    m = _PAT.match(name)
    return None if not m else (m.group(1), m.group(2), m.group(3), m.group(4))


def cluster_labels(r, comps):
    """Integer cluster label per stop (depot = -1) for the clustering by the components `comps` (indices into G, x, y, D)."""
    ids, labels = {}, np.full(len(r.zone), -1)
    for i in range(1, len(r.zone)):
        name = r.zone_names[r.zone[i]]; p = parse(name)
        key = tuple(p[k] for k in comps) if p else ("?", name)          # an unparsable ID is its own cluster
        labels[i] = ids.setdefault(key, len(ids))
    return labels


def excess_crossings(r, comps):
    """How many more cluster changes the driver made than the minimum (clusters - 1); 0 = every cluster visited in one block."""
    lab = cluster_labels(r, comps)[np.asarray(r.actual)[1:]]
    return int((lab[1:] != lab[:-1]).sum()) - (len(set(lab.tolist())) - 1)


def chains():
    """All nested chains (S, T, U): |S| = 3 > |T| = 2 > |U| = 1 components out of {G, x, y, D} = {0, 1, 2, 3}."""
    out = []
    for S in itertools.combinations(range(4), 3):
        for T in itertools.combinations(S, 2):
            for U in itertools.combinations(T, 1):
                out.append((S, T, U))
    return out


def best_chain(train_routes):
    """The chain whose three levels are crossed least by the training drivers (sum over levels of the mean excess crossings)."""
    scores = []
    for ch in chains():
        per = [float(np.mean([excess_crossings(r, c) for r in train_routes])) for c in ch]
        scores.append((sum(per), ch, per))
    scores.sort(key=lambda t: t[0])
    return scores


def save_chain(ch, path=None):
    path = path or config.CACHE_DIR / "hier_chain.json"; config.CACHE_DIR.mkdir(exist_ok=True)
    json.dump([list(c) for c in ch], open(path, "w"))


def load_chain(path=None):
    path = path or config.CACHE_DIR / "hier_chain.json"
    return [tuple(c) for c in json.load(open(path))]


def levels_for(r, weight, chain=None):
    """[(labels, weight)] for penalised_matrix: every level of the chain adds `weight` seconds for leaving a cluster."""
    chain = chain or load_chain()
    return [(cluster_labels(r, c), float(weight)) for c in chain] if weight > 0 else None
