import numpy as np
from decimal import Decimal
from lmr.data import build_route, split_of

def test_build_route_floats_and_order():
    nan = float("nan")
    meta = {"route_score": "High", "station_code": "DLA3", "stops": {
        "S": {"lat": 0, "lng": 0, "type": "Station", "zone_id": nan},
        "A": {"lat": 1, "lng": 1, "type": "Dropoff", "zone_id": "Z1"},
        "B": {"lat": 2, "lng": 2, "type": "Dropoff", "zone_id": "Z2"}}}
    seq = {"actual": {"S": 0, "A": 2, "B": 1}}
    mat = {a: {b: Decimal("1.5") for b in "SAB"} for a in "SAB"}      # ijson returns Decimal
    r = build_route("r", meta, seq, mat)
    assert r.T.dtype == np.float64 and r.zone[0] == -1
    assert r.ids == ["S", "B", "A"] and list(r.actual) == [0, 1, 2]

def test_split_is_stable():
    assert split_of("RouteID_abc") == split_of("RouteID_abc")