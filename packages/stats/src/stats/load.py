from pathlib import Path

from shared.dataset import load_dataset
from shared.models import DatasetRow


def load(path: str | Path) -> list[DatasetRow]:
    rows = load_dataset(path)
    return rows
