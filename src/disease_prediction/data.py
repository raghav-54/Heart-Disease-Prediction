from pathlib import Path

import pandas as pd


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load dataset from CSV."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    return pd.read_csv(data_path)

