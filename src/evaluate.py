import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay)
from preprocess import OUT


def evaluate(model, X_tr, X_te, y_tr, y_te):
    lines = []
    for name, Xs, ys in [("Train", X_tr, y_tr), ("Test", X_te, y_te)]:
        pred = model.predict(Xs)
        prob = model.predict_proba(Xs)[:, 1]
        lines.append("%s: accuracy %.3f | precision %.3f | recall %.3f | F1 %.3f | ROC-AUC %.3f" % (
            name, accuracy_score(ys, pred), precision_score(ys, pred),
            recall_score(ys, pred), f1_score(ys, pred), roc_auc_score(ys, prob)))
    print("\n=== Evaluation ===")
    print("\n".join(lines))
    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    ConfusionMatrixDisplay(confusion_matrix(y_te, model.predict(X_te))).plot()
    plt.title("Confusion matrix (test set)")
    plt.savefig(os.path.join(OUT, "confusion_matrix.png"))
    plt.close()
    RocCurveDisplay.from_estimator(model, X_te, y_te)
    plt.savefig(os.path.join(OUT, "roc_curve.png"))
    plt.close()
