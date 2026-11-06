import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from preprocess import OUT


def train(X_train, y_train, column_names):
    # train the model
    # class_weight="balanced" makes mistakes on the rare purchase class count more
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    # plot 1: the 8 most negative and 8 most positive coefficients
    coefs = pd.Series(model.coef_[0], index=column_names).sort_values()
    pd.concat([coefs.head(8), coefs.tail(8)]).plot.barh(figsize=(8, 6))
    plt.title("Largest coefficients")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "top_coefficients.png"))
    plt.close()

    # odds ratios: above 1 raises the odds of purchase, below 1 lowers them
    odds = np.exp(coefs).sort_values(ascending=False)
    odds.to_csv(os.path.join(OUT, "odds_ratios.csv"), header=["odds_ratio"])

    # plot 2: learning curve (score as the model sees more training data)
    sizes, train_scores, val_scores = learning_curve(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        X_train, y_train, cv=5, scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 5))
    plt.plot(sizes, train_scores.mean(axis=1), label="Train")
    plt.plot(sizes, val_scores.mean(axis=1), label="Validation")
    plt.xlabel("Number of training examples")
    plt.ylabel("ROC-AUC")
    plt.legend()
    plt.savefig(os.path.join(OUT, "learning_curve.png"))
    plt.close()

    return model