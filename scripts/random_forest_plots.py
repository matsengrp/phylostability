import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import math

plt.rcParams.update({"font.size": 12})  # Adjust this value as needed
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12


def plot_random_forest_results(results_csv, plot_filepath):
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


def plot_random_forest_model_features(model_features_csv, plot_filepath):
    df = pd.read_csv(
        model_features_csv, names=["feature_name", "importance"], skiprows=1
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="feature_name", y="importance")
    plt.title("feature importance for random forest regression")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def plot_tiis(csv, plot_filepath):
    df = pd.read_csv(csv)
    datasets = df["dataset"].unique()
    datasets.sort()
    if len(datasets) > 20:
        # Instead of individual TII plots, boxplot representing TIIs for all data sets
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(data=df, x="normalised_tii")
        # Set labels and title
        plt.xlabel("TII")
        plt.title("TII values over all datasets")
        plt.tight_layout()
        plt.savefig(plot_filepath)
        plt.clf()
        return 1
    num_datasets = len(datasets)
    num_rows = int(num_datasets**0.5)
    num_cols = int(num_datasets / num_rows) + (num_datasets % num_rows > 0)

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
        bin_start = math.floor(current_df["tii"].min()) - 0.5
        bin_end = math.ceil(current_df["tii"].max()) + 1 + 1.5

        # Plot histogram with binwidth=1 and adjusted binrange
        sns.histplot(
            data=current_df, x="tii", ax=ax, binwidth=1, binrange=(bin_start, bin_end)
        )
        ax.set_title(dataset + " (n = " + str(len(current_df)) + ")")
        # Set x-axis label only for bottom row plots
        if row == num_rows - 1:
            ax.set_xlabel("TII", fontsize=14)
        else:
            ax.set_xlabel("")

        # Set y-axis label only for leftmost column plots
        if col == 0:
            ax.set_ylabel("Frequency", fontsize=14)
        else:
            ax.set_ylabel("")

    # plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


results_csv = snakemake.input.random_forest_csv
model_features_csv = snakemake.input.model_features_csv
combined_csv = snakemake.input.combined_csv_path
plots_folder = snakemake.params.forest_plot_folder


if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

print("Start plotting TIIs.")
plot_filepath = os.path.join(plots_folder, "tii.pdf")
plot_tiis(combined_csv, plot_filepath)
print("Done plotting TIIs.")

print("Start plotting random forest results.")
random_forest_plot_filepath = os.path.join(plots_folder, "random_forest_results.pdf")
plot_random_forest_results(results_csv, random_forest_plot_filepath)
model_features_plot_filepath = os.path.join(
    plots_folder, "random_forest_model_features.pdf"
)
plot_random_forest_model_features(model_features_csv, model_features_plot_filepath)
print("Done plotting random forest results.")
