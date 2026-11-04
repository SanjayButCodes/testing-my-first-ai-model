import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# folder paths (work from any location)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE, "data", "online_shoppers_intention.csv")
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)


def load_raw():
    return pd.read_csv(DATA_FILE)


def get_data():
    df = load_raw()

    # 1. remove repeated rows
    df = df.drop_duplicates()

    # 2. turn True/False into 1/0
    df["Revenue"] = df["Revenue"].astype(int)
    df["Weekend"] = df["Weekend"].astype(int)

    # 3. one-hot encode the category columns
    category_cols = ["Month", "VisitorType", "OperatingSystems",
                     "Browser", "Region", "TrafficType"]
    df = pd.get_dummies(df, columns=category_cols, drop_first=True)

    # 4. separate features (X) and target (y)
    y = df["Revenue"]
    X = df.drop(columns="Revenue")

    # 5. split into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 6. scale features (learn the scaling from the training data only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, list(X.columns)