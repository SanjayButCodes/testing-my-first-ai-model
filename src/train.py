import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict, learning_curve
from sklearn.metrics import precision_recall_curve
from preprocess import OUT


def train(X_train, y_train, column_names):
    # 1. try a few values of C (regularization strength) with 5-fold cross-validation
    #    and keep the one with the best ROC-AUC
    search = GridSearchCV(LogisticRegression(max_iter=1000),
                          {"C": [0.01, 0.1, 1, 10]}, cv=5, scoring="roc_auc")
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print("Best C found by cross-validation:", search.best_params_["C"])

    # 2. choose the decision threshold that gives the best F1 score
    #    (predictions come from cross-validation, so we never use the test set)
    cv_probs = cross_val_predict(model, X_train, y_train, cv=5,
                                 method="predict_proba")[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, cv_probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best = np.argmax(f1[:-1])
    threshold = thresholds[best]
    print("Chosen decision threshold: %.3f" % threshold)

    # plot 1: precision and recall as the threshold changes
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.plot(thresholds, recall[:-1], label="Recall")
    plt.axvline(threshold, color="gray", linestyle="--", label="Chosen threshold")
    plt.xlabel("Threshold")
    plt.legend()
    plt.title("Precision and recall vs threshold (training data)")
    plt.savefig(os.path.join(OUT, "threshold_curve.png"))
    plt.close()

    # plot 2: the 8 most negative and 8 most positive coefficients
    coefs = pd.Series(model.coef_[0], index=column_names).sort_values()
    pd.concat([coefs.head(8), coefs.tail(8)]).plot.barh(figsize=(8, 6))
    plt.title("Largest coefficients")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "top_coefficients.png"))
    plt.close()

    # odds ratios: above 1 raises the odds of purchase, below 1 lowers them
    odds = np.exp(coefs).sort_values(ascending=False)
    odds.to_csv(os.path.join(OUT, "odds_ratios.csv"), header=["odds_ratio"])

    # plot 3: learning curve (score as the model sees more training data)
    sizes, train_scores, val_scores = learning_curve(
        LogisticRegression(C=search.best_params_["C"], max_iter=1000),
        X_train, y_train, cv=5, scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 5))
    plt.plot(sizes, train_scores.mean(axis=1), label="Train")
    plt.plot(sizes, val_scores.mean(axis=1), label="Validation")
    plt.xlabel("Number of training examples")
    plt.ylabel("ROC-AUC")
    plt.legend()
    plt.savefig(os.path.join(OUT, "learning_curve.png"))
    plt.close()

    # plot 4: predicted purchase probability for each class
    probs = model.predict_proba(X_train)[:, 1]
    plt.hist(probs[y_train.values == 0], bins=30, alpha=0.6, label="No purchase")
    plt.hist(probs[y_train.values == 1], bins=30, alpha=0.6, label="Purchase")
    plt.xlabel("Predicted probability of purchase")
    plt.legend()
    plt.savefig(os.path.join(OUT, "probability_histogram.png"))
    plt.close()

    return model, threshold