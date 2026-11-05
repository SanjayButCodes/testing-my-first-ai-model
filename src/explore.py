import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preprocess import load_raw, OUT

# these columns hold anonymous category codes, so correlations with them mean little
CODE_COLS = ["OperatingSystems", "Browser", "Region", "TrafficType"]


def run():
    df = load_raw()
    numbers = df.drop(columns=CODE_COLS).select_dtypes(include="number")

    print("=== Mean ===")
    print(numbers.mean())
    print("\n=== Median ===")
    print(numbers.median())
    print("\n=== Standard deviation ===")
    print(numbers.std())
    print("\n=== Covariance ===")
    print(numbers.cov())

    corr = numbers.corr()
    print("\n=== Correlation ===")
    print(corr)

    print("\n=== Correlation of each column with Revenue (purchase) ===")
    print(numbers.corrwith(df["Revenue"].astype(int)).sort_values(ascending=False))

    print("\n=== Class balance (Revenue) ===")
    print(df["Revenue"].value_counts(normalize=True))

    # correlation heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm")
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title("Correlation heatmap (numeric columns)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "correlation_heatmap.png"))
    plt.close()