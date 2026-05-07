import os

os.environ.setdefault("TRIED_ROLE", "stats")

from stats.config import DATASET_PATH  # noqa: E402
from stats.load import load  # noqa: E402
from stats.report import report  # noqa: E402

report(load(DATASET_PATH))
