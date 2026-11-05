import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# folder paths (work from any location)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE, "data", "online_shoppers_intention.csv")
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

# columns with a long right tail: we apply log(1 + x) to squeeze them
SKEWED_COLS = ["Administrative", "Administrative_Duration",
               "Informational", "Informational_Duration",
               "ProductRelated", "ProductRelated_Duration", "PageValues"]

# code columns: keep only the N most common codes, call the rest "Other"
KEEP_TOP = {"OperatingSystems": 3, "Browser": 3, "TrafficType": 3}


def load_raw():
    return pd.read_csv(DATA_FILE)


def get_data():
    df = load_raw()

    # 1. remove repeated rows
    df = df.drop_duplicates()

    # 2. turn True/False into 1/0
    df["Revenue"] = df["Revenue"].astype(int)
    df["Weekend"] = df["Weekend"].astype(int)

    # 3. drop columns we do not need
    #    Region: anonymous code with little meaning
    #    BounceRates: almost the same information as ExitRates
    df = df.drop(columns=["Region", "BounceRates"])

    # 4. separate features (X) and target (y)
    y = df["Revenue"]
    X = df.drop(columns="Revenue")

    # 5. split into train (80%) and test (20%) BEFORE any learning from the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_train = X_train.copy()
    X_test = X_test.copy()

    # 6. group rare categories into "Other"
    #    the most common codes are found from the training data only
    for col, n in KEEP_TOP.items():
        common = X_train[col].value_counts().head(n).index
        X_train[col] = X_train[col].where(X_train[col].isin(common), "Other").astype(str)
        X_test[col] = X_test[col].where(X_test[col].isin(common), "Other").astype(str)

    # 7. log transform the skewed columns
    X_train[SKEWED_COLS] = np.log1p(X_train[SKEWED_COLS])
    X_test[SKEWED_COLS] = np.log1p(X_test[SKEWED_COLS])

    # 8. one-hot encode the category columns
    category_cols = ["Month", "VisitorType"] + list(KEEP_TOP)
    X_train = pd.get_dummies(X_train, columns=category_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=category_cols, drop_first=True)
    # make sure test has exactly the same columns, in the same order, as train
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    column_names = list(X_train.columns)

    # 9. scale features (learn the scaling from the training data only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.astype(float))
    X_test = scaler.transform(X_test.astype(float))

    return X_train, X_test, y_train, y_test, column_names