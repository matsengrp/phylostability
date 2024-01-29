import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import confusion_matrix, auc, ConfusionMatrixDisplay


dark2 = mpl.colormaps["Dark2"]


def plot_random_forest_classifier_results(results_csv, roc_csv, plot_filepath):
    df = (
        pd.read_csv(results_csv)
        .replace(to_replace=0, value="unstable")
        .replace(to_replace=1, value="stable")
    )
    cm = confusion_matrix(df["actual"], df["predicted"])

    roc_df = pd.read_csv(roc_csv)
    roc_auc = auc(roc_df["fpr"], roc_df["tpr"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Colours
    ax1.plot(
        roc_df["fpr"],
        roc_df["tpr"],
        color=dark2.colors[0],
        lw=2,
        label="ROC curve (area={:.2f})".format(roc_auc),
    )
    ax1.plot([0, 1], [0, 1], color=dark2.colors[1], lw=2, linestyle="--")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax2, cmap="Blues")
    ax2.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def plot_au_test_pie_chart(df, plot_filepath):
    # Pie Chart
    condition1 = ((df["normalised_tii"] == 0.0) & (df["p-AU"] < 0.05)).sum()
    condition2 = ((df["normalised_tii"] > 0.0) & (df["p-AU"] < 0.05)).sum()
    condition3 = ((df["normalised_tii"] == 0.0) & (df["p-AU"] >= 0.05)).sum()
    condition4 = ((df["normalised_tii"] > 0.0) & (df["p-AU"] >= 0.05)).sum()

    # Data to plot
    sizes = [condition1, condition2, condition3, condition4]
    total = sum(sizes)
    percentages = [100 * (size / total) for size in sizes]
    labels = [
        "stable and significant ({:.1f}%)".format(percentages[0]),
        "unstable and significant ({:.1f}%)".format(percentages[1]),
        "stable and non-significant ({:.1f}%)".format(percentages[2]),
        "unstable and non-significant ({:.1f}%)".format(percentages[3]),
    ]
    # Colors
    colors = [dark2.colors[i] for i in range(3, -1, -1)]
    # Exploding the 1st slice (optional)
    explode = (0.1, 0, 0, 0)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        # autopct="%1.1f%%",
        # shadow=True,
        startangle=140,
    )
    plt.axis("equal")
    plt.savefig(plot_filepath)


classifier_results = snakemake.input.classifier_results
classifier_metrics_csv = snakemake.input.classifier_metrics_csv
classifier_plot_file = snakemake.output.classifier_plot_file
all_au_test_results = snakemake.input.all_au_test_results
pie_plot_file = snakemake.output.pie_plot_file

plot_random_forest_classifier_results(
    classifier_results, classifier_metrics_csv, classifier_plot_file
)

df = pd.read_csv(all_au_test_results)
plot_au_test_pie_chart(df, pie_plot_file)
