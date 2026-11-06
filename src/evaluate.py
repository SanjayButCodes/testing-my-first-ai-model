import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay)
from preprocess import OUT


def show_metrics(name, model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return "%s: accuracy %.3f | precision %.3f | recall %.3f | F1 %.3f | ROC-AUC %.3f" % (
        name,
        accuracy_score(y, predictions),
        precision_score(y, predictions),
        recall_score(y, predictions),
        f1_score(y, predictions),
        roc_auc_score(y, probabilities))


def evaluate(model, X_train, X_test, y_train, y_test):
    train_line = show_metrics("Train", model, X_train, y_train)
    test_line = show_metrics("Test", model, X_test, y_test)
    baseline = "Baseline (always predict no purchase): accuracy %.3f" % (1 - y_test.mean())

    print("\n=== Evaluation ===")
    print(train_line)
    print(test_line)
    print(baseline)

    # save the numbers to a text file
    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write(train_line + "\n" + test_line + "\n" + baseline + "\n")

    # confusion matrix (test set)
    matrix = confusion_matrix(y_test, model.predict(X_test))
    ConfusionMatrixDisplay(matrix).plot()
    plt.title("Confusion matrix (test set)")
    plt.savefig(os.path.join(OUT, "confusion_matrix.png"))
    plt.close()

    # ROC curve (test set)
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.savefig(os.path.join(OUT, "roc_curve.png"))
    plt.close()