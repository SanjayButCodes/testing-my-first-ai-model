import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preprocess import load_raw, OUT


def run():
    df = load_raw()
    num = df.select_dtypes(include="number")
    stats = num.describe().T[["mean", "std"]].assign(median=num.median())
    print("=== Mean, std, median ===")
    print(stats)
    print("\n=== Covariance ===")
    print(num.cov())
    corr = num.corr()
    print("\n=== Correlation ===")
    print(corr)
    print("\n=== Class balance (Revenue) ===")
    print(df["Revenue"].value_counts(normalize=True))

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm")
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "correlation_heatmap.png"))
    plt.close()
