import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, auc, ConfusionMatrixDisplay


def plot_random_forest_classifier_results(results_csv, roc_csv, plot_filepath):
    df = (
        pd.read_csv(results_csv)
        .replace(to_replace=0, value="unstable")
        .replace(to_replace=1, value="stable")
    )
    print(df)
    cm = confusion_matrix(df["actual"], df["predicted"])

    roc_df = pd.read_csv(roc_csv)
    roc_auc = auc(roc_df["fpr"], roc_df["tpr"])

    print(roc_auc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(
        roc_df["fpr"],
        roc_df["tpr"],
        color="darkorange",
        lw=2,
        label="ROC curve (area={:.2f})".format(roc_auc),
    )
    ax1.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax2)
    ax2.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


classifier_results = snakemake.input.classifier_results
classifier_metrics_csv = snakemake.input.classifier_metrics_csv
plot_file = snakemake.output.plot_file

plot_random_forest_classifier_results(
    classifier_results, classifier_metrics_csv, plot_file
)
