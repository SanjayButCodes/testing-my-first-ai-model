import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data", "online_shoppers_intention.csv")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)


def load_raw():
    return pd.read_csv(DATA)


def get_data():
    df = load_raw().drop_duplicates()
    df["Revenue"] = df["Revenue"].astype(int)
    df["Weekend"] = df["Weekend"].astype(int)
    cat = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]
    df = pd.get_dummies(df, columns=cat, drop_first=True)
    X, y = df.drop(columns="Revenue"), df["Revenue"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te, list(X.columns)
