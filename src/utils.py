from pathlib import Path
import pandas as pd


def ensure_directories(paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_csv(path):
    return pd.read_csv(path)


def describe_dataframe(df):
    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": df.columns.tolist(),
        "missing_values": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "duplicated_rows": int(df.duplicated().sum()),
    }