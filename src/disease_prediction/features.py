import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering used in the notebook."""
    out = df.copy()
    out["age_group"] = pd.cut(out["age"], bins=[0, 40, 55, 100], labels=[0, 1, 2]).astype(int)
    out["high_chol"] = (out["chol"] > 240).astype(int)
    out["stress_index"] = out["oldpeak"] * out["thalach"]
    out["age_chol"] = out["age"] * out["chol"]
    return out

