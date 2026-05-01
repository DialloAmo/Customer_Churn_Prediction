import numpy as np
import pandas as pd

SERVICES_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]

def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop("customerID", axis=1, errors="ignore")
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    return df

def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, float("inf")],
        labels=["nouveaux", "intermediaires", "fideles"],
        right=False
    )
    return df

def add_num_services(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["num_services"] = (df[SERVICES_COLS] == "Yes").sum(axis=1)
    return df

def add_engagement_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    engagement_score = (
        df["Contract"].isin(["One year", "Two year"]).astype(int)
        + (df["num_services"] >= 3).astype(int)
        + df["InternetService"].isin(["DSL", "Fiber optic"]).astype(int)
        + df["PaymentMethod"].isin([
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]).astype(int)
    )

    df["engagement_level"] = pd.cut(
        engagement_score,
        bins=[-1, 1, 3, 4],
        labels=["faible", "moyen", "eleve"]
    )
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_telco_data(df)
    df = add_tenure_group(df)
    df = add_num_services(df)
    df = add_engagement_level(df)
    return df