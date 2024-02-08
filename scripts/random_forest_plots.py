import warnings

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")
warnings.filterwarnings("ignore", "UserWarning")
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import math
from sklearn.metrics import confusion_matrix, auc, ConfusionMatrixDisplay

# Font sizes for figures
plt.rcParams.update({"font.size": 12})  # Adjust this value as needed
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16

# Colour for plots
dark2 = mpl.colormaps["Dark2"]


def plot_random_forest_regression_results(
    results_csv, plot_filepath, stability_measure, r2_file
):
    with open(r2_file, "r") as f:
        r2 = float(f.readlines()[0].strip())
    df = pd.read_csv(results_csv)
    df_sorted = df.sort_values(by="actual")

    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(
        data=df_sorted, x="actual", y="predicted", color=dark2.colors[0]
    )

    # Determine the common maximum limit for both axes
    common_limit = max(df["actual"].max(), df["predicted"].max()) + 0.01

    textstr = f"R²= {r2:.2f}"  # Formats the string to display R² with 2 decimal places
    props = dict(boxstyle="square", facecolor="white", alpha=0.5)

    # Position: x, y, text, properties of the box
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    # Set the same limits for both the x-axis and y-axis
    plt.xlim(-0.01, common_limit)
    plt.ylim(-0.01, common_limit)

    plt.title("")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def plot_random_forest_classifier_results(
    results_csv, roc_csv, plot_filepath, subplots=False
):
    if subplots == True:
        df = (
            pd.read_csv(results_csv)
            .replace(to_replace=True, value="unstable")
            .replace(to_replace=False, value="stable")
        )
        conf_m = confusion_matrix(df["actual"], df["predicted"])

        roc_df = pd.read_csv(roc_csv)
        roc_auc = auc(roc_df["fpr"], roc_df["tpr"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

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
        ax1.set_title("")
        ax1.legend(loc="lower right")
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_m)
        disp.plot(ax=ax2, cmap="Blues", colorbar=False)
        ax2.set_title("")
        plt.tight_layout()
        plt.savefig(plot_filepath)
        plt.clf()
    else:
        df = (
            pd.read_csv(results_csv)
            .replace(to_replace=True, value="unstable")
            .replace(to_replace=False, value="stable")
        )
        conf_m = confusion_matrix(df["actual"], df["predicted"])

        roc_df = pd.read_csv(roc_csv)
        roc_auc = auc(roc_df["fpr"], roc_df["tpr"])

        plt.figure(figsize=(6, 6))

        plt.plot(
            roc_df["fpr"],
            roc_df["tpr"],
            color=dark2.colors[0],
            lw=2,
            label="ROC curve (area={:.2f})".format(roc_auc),
        )
        plt.plot([0, 1], [0, 1], color=dark2.colors[1], lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plot_filepath)
        plt.clf()


def plot_random_forest_model_features(model_features_csv, plot_filepath):
    df = pd.read_csv(
        model_features_csv,
        names=["feature_name", "untuned model importance", "importance"],
        skiprows=1,
    )
    feature_name_dict = {
        "num_leaves": "#leaves",
        "like_weight_ratio": "LWR",
        "reattachment_branch_length": "insertion branch length",
        "distal_length": "distal length",
        "pendant_branch_length": "pendant length",
        "taxon_height": "insertion height",
        "num_likely_reattachments": "#insertion locations",
        "nj_tii": "NJ TII",
        "dist_reattachment_low_bootstrap_node": "distance to low bootstrap node",
        "dist_diff_reattachment_sibling": "dist diff insertion sibling",
        "bootstrap_mean": "bootstrap mean",
        "bootstrap_std": "bootstrap SD",
        "reattachment_distances_mean": "dist to insertion mean",
        "reattachment_distances_std": "dist to insertion SD",
        "seq_and_tree_dist_ratio_mean": "distance ratio mean",
        "seq_and_tree_dist_ratio_std": "distance ratio SD",
        "seq_distance_ratios_closest_seq_mean": "ratio diff closest sequence mean",
        "seq_distance_ratios_closest_seq_std": "ratio diff closest sequence SD",
    }
    df["new_feature_name"] = list(
        map(lambda x: feature_name_dict[x], df["feature_name"])
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="new_feature_name", y="importance", color=dark2.colors[0])
    plt.xticks(rotation=90)
    plt.title("")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def plot_combined_random_forest_model_features(
    classification_features_csv, regression_features, plot_filepath
):
    feature_name_dict = {
        "num_leaves": "#leaves",
        "like_weight_ratio": "LWR",
        "reattachment_branch_length": "insertion branch length",
        "distal_length": "distal length",
        "pendant_branch_length": "pendant length",
        "taxon_height": "insertion height",
        "num_likely_reattachments": "#insertion locations",
        "nj_tii": "NJ TII",
        "dist_reattachment_low_bootstrap_node": "distance to low bootstrap node",
        "dist_diff_reattachment_sibling": "dist diff insertion sibling",
        "bootstrap_mean": "bootstrap mean",
        "bootstrap_std": "bootstrap SD",
        "reattachment_distances_mean": "dist to insertion mean",
        "reattachment_distances_std": "dist to insertion SD",
        "seq_and_tree_dist_ratio_mean": "distance ratio mean",
        "seq_and_tree_dist_ratio_std": "distance ratio SD",
        "seq_distance_ratios_closest_seq_mean": "ratio diff closest sequence mean",
        "seq_distance_ratios_closest_seq_std": "ratio diff closest sequence SD",
    }
    classification_df = pd.read_csv(
        classification_features_csv,
        names=["feature_name", "untuned model importance", "importance"],
        skiprows=1,
    )
    classification_df["new_feature_name"] = list(
        map(lambda x: feature_name_dict[x], classification_df["feature_name"])
    )
    classification_df["type"] = "classification"

    regression_df = pd.read_csv(
        regression_features,
        names=["feature_name", "untuned model importance", "importance"],
        skiprows=1,
    )
    regression_df["new_feature_name"] = list(
        map(lambda x: feature_name_dict[x], regression_df["feature_name"])
    )
    regression_df["type"] = "regression"
    df = pd.concat([classification_df, regression_df])

    plt.figure(figsize=(10, 6))
    palette = [dark2.colors[0], dark2.colors[1]]
    sns.barplot(
        data=df, x="new_feature_name", y="importance", hue="type", palette=palette
    )
    plt.xticks(rotation=90)
    plt.title("")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def plot_stability_measures(
    csv,
    rf_radius_plot_filepath,
    tii_plot_filepath,
    normalised_tii_plot_filepath,
    scatterplot_filepath,
    rf_radius_num_taxa_filepath,
    plot_individual=False,
):
    df = pd.read_csv(csv)
    datasets = df["dataset"].unique()
    datasets.sort()
    # Scatter plot with point color indicating count
    aggregated_data = (
        df.groupby(["normalised_tii", "rf_radius"]).size().reset_index(name="counts")
    )
    sns.scatterplot(
        data=aggregated_data,
        x="normalised_tii",
        y="rf_radius",
        hue="counts",
        color=dark2.colors[0],
    )

    # Set labels and title
    plt.xlabel("TII")
    plt.ylabel("RF radius")
    plt.title("")
    plt.tight_layout()
    plt.savefig(scatterplot_filepath)
    plt.clf()

    # plot RF radius vs num_leaves
    sns.scatterplot(
        data=df,
        x="num_leaves",
        y="rf_radius",
        color=dark2.colors[0]
    )
    plt.xlabel("Number of taxa")
    plt.ylabel("RF radius")
    plt.title("")
    plt.tight_layout()
    plt.savefig(rf_radius_num_taxa_filepath)
    plt.clf()

    if not plot_individual:
        # Instead of individual TII plots, histogram representing stability measures for all data sets
        # plt.figure(figsize=(10, 6))
        # Plot RF radius
        max_radius = max(df["rf_radius"])
        num_bins = len(df["rf_radius"].unique())
        bins = [(i - 0.5) * max_radius / num_bins for i in range(0, num_bins)]
        sns.histplot(data=df, x="rf_radius", bins=bins, color=dark2.colors[0])
        # Set labels and title
        plt.xlabel("RF radius")
        plt.title("")
        plt.tight_layout()
        plt.savefig(rf_radius_plot_filepath)
        plt.clf()
        # plot TII
        max_tii = max(df["tii"])
        num_bins = 100  # len(df["normalised_tii"].unique())
        bins = [(i - 0.5) * max_tii / num_bins for i in range(0, num_bins)]
        sns.histplot(data=df, x="tii", bins=bins, color=dark2.colors[0])
        # Set labels and title
        plt.xlabel("TII")
        plt.title("")
        plt.tight_layout()
        plt.savefig(tii_plot_filepath)
        plt.clf()
        # plot normalised TII
        max_tii = max(df["normalised_tii"])
        num_bins = 100  # len(df["normalised_tii"].unique())
        bins = [(i - 0.5) * max_tii / num_bins for i in range(0, num_bins)]
        sns.histplot(data=df, x="normalised_tii", bins=bins, color=dark2.colors[0])
        # Set labels and title
        plt.xlabel("TII")
        plt.title("")
        plt.tight_layout()
        plt.savefig(normalised_tii_plot_filepath)
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
                data=current_df,
                x=var,
                ax=ax,
                binwidth=1,
                binrange=(bin_start, bin_end),
                color=dark2.colors[0],
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
r2_file = snakemake.input.r2_file
au_test_classifier_results = snakemake.input.au_test_classifier_results
au_test_classifier_metrics_csv = snakemake.input.au_test_classifier_metrics_csv
au_test_features = snakemake.input.au_test_model_features_file
plots_folder = snakemake.params.forest_plot_folder
stability_measure = snakemake.params.stability_measure


if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

print("Start plotting stability measures.")
tii_plot_filepath = os.path.join(plots_folder, "tii.pdf")
normalised_tii_plot_filepath = os.path.join(plots_folder, "normalised_tii.pdf")
rf_radius_plot_filepath = os.path.join(plots_folder, "rf_radius.pdf")
scatterplot_filepath = os.path.join(plots_folder, "tii_vs_rf_radius.pdf")
rf_radius_num_taxa_filepath = os.path.join(plots_folder, "rf_radius_num_taxa.pdf")
plot_stability_measures(
    combined_csv,
    rf_radius_plot_filepath,
    tii_plot_filepath,
    normalised_tii_plot_filepath,
    scatterplot_filepath,
    rf_radius_num_taxa_filepath,
)
print("Done plotting stability measures.")

print("Start plotting random forest regression results.")
random_forest_plot_filepath = os.path.join(plots_folder, "random_forest_results.pdf")
plot_random_forest_regression_results(
    results_csv, random_forest_plot_filepath, stability_measure, r2_file
)
print("Done plotting random forest regresion results.")

print("Start plotting random forest classifier results.")
random_forest_plot_filepath = os.path.join(
    plots_folder, "random_forest_classifier_results.pdf"
)
plot_random_forest_classifier_results(
    classifier_results_csv, classifier_metrics_csv, random_forest_plot_filepath
)
print("Done plotting random forest classifier results.")

print("Start plotting au test classifier results")
filepath = os.path.join(plots_folder, "au_test_classifier_results.pdf")
plot_random_forest_classifier_results(
    au_test_classifier_results, au_test_classifier_metrics_csv, filepath
)
print("Done plotting au test classifier results")


print("Start plotting feature importances.")
plot_filepath = os.path.join(plots_folder, "au_test_classifier_features.pdf")
plot_random_forest_model_features(au_test_features, plot_filepath)

model_features_plot_filepath = os.path.join(
    plots_folder, "random_forest_model_features.pdf"
)
plot_random_forest_model_features(model_features_csv, model_features_plot_filepath)
model_features_plot_filepath = os.path.join(
    plots_folder, "discrete_random_forest_model_features.pdf"
)
plot_random_forest_model_features(
    discrete_model_features_csv, model_features_plot_filepath
)
plot_filepath = os.path.join(plots_folder, "combined_random_forest_features.pdf")
plot_combined_random_forest_model_features(
    discrete_model_features_csv, model_features_csv, plot_filepath
)
print("Done plotting feature importances.")
