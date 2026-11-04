import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from preprocess import OUT


def train(X_tr, y_tr, cols):
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_tr, y_tr)

    # top coefficients
    coef = pd.Series(model.coef_[0], index=cols).sort_values()
    pd.concat([coef.head(8), coef.tail(8)]).plot.barh(figsize=(8, 6))
    plt.title("Largest coefficients (scaled features)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "top_coefficients.png"))
    plt.close()

    # odds ratios for the analysis
    odds = np.exp(coef).sort_values(ascending=False)
    odds.to_csv(os.path.join(OUT, "odds_ratios.csv"), header=["odds_ratio"])

    # learning curve
    sizes, tr_sc, va_sc = learning_curve(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        X_tr, y_tr, cv=5, scoring="f1", train_sizes=np.linspace(0.1, 1.0, 5))
    plt.plot(sizes, tr_sc.mean(axis=1), label="Train")
    plt.plot(sizes, va_sc.mean(axis=1), label="Validation")
    plt.xlabel("Training examples")
    plt.ylabel("F1 score")
    plt.legend()
    plt.savefig(os.path.join(OUT, "learning_curve.png"))
    plt.close()

    # predicted probabilities by class
    prob = model.predict_proba(X_tr)[:, 1]
    plt.hist(prob[y_tr.values == 0], bins=30, alpha=0.6, label="No purchase")
    plt.hist(prob[y_tr.values == 1], bins=30, alpha=0.6, label="Purchase")
    plt.xlabel("Predicted purchase probability")
    plt.legend()
    plt.savefig(os.path.join(OUT, "probability_histogram.png"))
    plt.close()
    return model
