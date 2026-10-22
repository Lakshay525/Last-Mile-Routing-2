import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("LMR_DATA_ROOT", "D:/Deakin University/Dataset/almrrc2021"))
BUILD = DATA_ROOT / "almrrc2021-data-training" / "model_build_inputs"
CACHE_DIR = Path(os.environ.get("LMR_CACHE", ".cache"))

SEED = 0
SA_ITERS = 200_000
LNS_ITERS = 1500
ZONE_SA_ITERS = 20_000
ZONE_LNS_ITERS = 300