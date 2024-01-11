import warnings

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")
warnings.filterwarnings("ignore", "UserWarning")
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
from sklearn.metrics import confusion_matrix, auc, ConfusionMatrixDisplay

plt.rcParams.update({"font.size": 12})  # Adjust this value as needed
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12


def plot_random_forest_regression_results(results_csv, plot_filepath):
    df = pd.read_csv(results_csv)
    df_sorted = df.sort_values(by="actual")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_sorted, x="actual", y="predicted")

    # Determine the common maximum limit for both axes
    common_limit = max(df["actual"].max(), df["predicted"].max()) + 0.01

    # Set the same limits for both the x-axis and y-axis
    plt.xlim(-0.01, common_limit)
    plt.ylim(-0.01, common_limit)

    plt.title("Results of Random Forest Regression")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def plot_random_forest_classifier_results(results_csv, roc_csv, plot_filepath):
    df = (
        pd.read_csv(results_csv)
        .replace(to_replace=True, value="unstable")
        .replace(to_replace=False, value="stable")
    )
    cm = confusion_matrix(df["actual"], df["predicted"])

    roc_df = pd.read_csv(roc_csv)
    roc_auc = auc(roc_df["fpr"], roc_df["tpr"])

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


def plot_random_forest_model_features(
    model_features_csv, plot_filepath, rf_type="regression"
):
    df = pd.read_csv(
        model_features_csv,
        names=["feature_name", "untuned model importance", "importance"],
        skiprows=1,
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="feature_name", y="importance")
    plt.title("feature importance for random forest " + rf_type)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def plot_stability_measures(
    csv, rf_radius_plot_filepath, tii_plot_filepath, scatterplot_filepath
):
    df = pd.read_csv(csv)
    datasets = df["dataset"].unique()
    datasets.sort()
    # Scatter plot with point color indicating count
    aggregated_data = df.groupby(["tii", "rf_radius"]).size().reset_index(name="counts")
    sns.scatterplot(
        data=aggregated_data, x="tii", y="rf_radius", hue="counts", palette="viridis"
    )

    # Set labels and title
    plt.xlabel("TII")
    plt.ylabel("RF radius")
    plt.title("Scatter plot with point size indicating count")
    plt.tight_layout()
    plt.savefig(scatterplot_filepath)
    plt.clf()

    if len(datasets) > 20:
        # Instead of individual TII plots, boxplot representing TIIs for all data sets
        plt.figure(figsize=(10, 6))
        # Plot RF radius
        max_radius = max(df["rf_radius"])
        num_bins = len(df["rf_radius"].unique())
        bins = [(i - 0.5) * max_radius / num_bins for i in range(0, num_bins)]
        sns.histplot(data=df, x="rf_radius", bins=bins)
        # Set labels and title
        plt.xlabel("RF radius")
        plt.title("RF radius over all datasets")
        plt.tight_layout()
        plt.savefig(rf_radius_plot_filepath)
        plt.clf()
        # plot TII
        max_radius = max(df["normalised_tii"])
        num_bins = len(df["normalised_tii"].unique())
        bins = [(i - 0.5) * max_radius / num_bins for i in range(0, num_bins)]
        sns.histplot(data=df, x="normalised_tii", bins=bins)
        # Set labels and title
        plt.xlabel("TII")
        plt.title("TII values over all datasets")
        plt.tight_layout()
        plt.savefig(tii_plot_filepath)
        plt.clf()
        return 1
    num_datasets = len(datasets)
    num_rows = int(num_datasets**0.5)
    num_cols = int(num_datasets / num_rows) + (num_datasets % num_rows > 0)

    for var in ["rf_radius", "tii"]:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 6))

        # Ensure that `axes` is always a 2D array
        if num_datasets == 1:
            axes = [[axes]]
        elif num_datasets in [2, 3]:
            axes = [axes]

        for index, dataset in enumerate(datasets):
            current_df = df.loc[df["dataset"] == dataset]
            row = index // num_cols
            col = index % num_cols
            ax = axes[row][col]
            # Your plotting code here using ax
            # Hide unused subplots
            if index == num_datasets - 1:
                for i in range(index + 1, num_rows * num_cols):
                    fig.delaxes(axes[i // num_cols][i % num_cols])
            # Calculate the appropriate bin range
            bin_start = math.floor(current_df[var].min()) - 0.5
            bin_end = math.ceil(current_df[var].max()) + 1 + 1.5

            # Plot histogram with binwidth=1 and adjusted binrange
            sns.histplot(
                data=current_df, x=var, ax=ax, binwidth=1, binrange=(bin_start, bin_end)
            )
            ax.set_title(dataset + " (n = " + str(len(current_df)) + ")")
            # Set x-axis label only for bottom row plots
            if row == num_rows - 1:
                ax.set_xlabel(var, fontsize=14)
            else:
                ax.set_xlabel("")

            # Set y-axis label only for leftmost column plots
            if col == 0:
                ax.set_ylabel("Frequency", fontsize=14)
            else:
                ax.set_ylabel("")

        # plt.tight_layout()
        if var == "rf_radius":
            plt.savefig(rf_radius_plot_filepath)
        else:
            plt.savefig(tii_plot_filepath)
        plt.clf()


results_csv = snakemake.input.random_forest_csv
model_features_csv = snakemake.input.model_features_csv
classifier_results_csv = snakemake.input.random_forest_classifier_csv
classifier_metrics_csv = snakemake.input.classifier_metrics_csv
discrete_model_features_csv = snakemake.input.discrete_model_features_csv
combined_csv = snakemake.input.combined_csv_path
plots_folder = snakemake.params.forest_plot_folder


if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

print("Start plotting stability measures.")
tii_plot_filepath = os.path.join(plots_folder, "tii.pdf")
rf_radius_plot_filepath = os.path.join(plots_folder, "rf_radius.pdf")
scatterplot_filepath = os.path.join(plots_folder, "tii_vs_rf_radius.pdf")
plot_stability_measures(
    combined_csv, rf_radius_plot_filepath, tii_plot_filepath, scatterplot_filepath
)
print("Done plotting stability measures.")


print("Start plotting random forest regression results.")
random_forest_plot_filepath = os.path.join(plots_folder, "random_forest_results.pdf")
plot_random_forest_regression_results(results_csv, random_forest_plot_filepath)
model_features_plot_filepath = os.path.join(
    plots_folder, "random_forest_model_features.pdf"
)
plot_random_forest_model_features(model_features_csv, model_features_plot_filepath)
print("Done plotting random forest regresion results.")

print("Start plotting random forest classifier results.")
random_forest_plot_filepath = os.path.join(
    plots_folder, "random_forest_classifier_results.pdf"
)
plot_random_forest_classifier_results(
    classifier_results_csv, classifier_metrics_csv, random_forest_plot_filepath
)
model_features_plot_filepath = os.path.join(
    plots_folder, "discrete_random_forest_model_features.pdf"
)
plot_random_forest_model_features(
    discrete_model_features_csv, model_features_plot_filepath, rf_type="classifier"
)
print("Done plotting random forest classifier results.")
