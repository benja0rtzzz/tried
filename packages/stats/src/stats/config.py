import os
from pathlib import Path

DATASET_PATH = Path(os.getenv("TRIED_DATASET", "data/dataset.jsonl"))
