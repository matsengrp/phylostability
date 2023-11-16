import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_random_forest_results(results_csv, plot_filepath):
    df = pd.read_csv(results_csv)
    df_sorted = df.sort_values(by="actual")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_sorted, x="actual", y="predicted")
    plt.title("results of random forest regression")
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
    num_datasets = len(datasets)
    num_rows = int(num_datasets**0.5)
    num_cols = int(num_datasets / num_rows) + (num_datasets % num_rows > 0)

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(15, 15)  # , sharex=True, sharey=True
    )
    for index, dataset in enumerate(datasets):
        row = index // num_cols
        col = index % num_cols
        ax = axes[index // num_cols, index % num_cols]
        current_df = df.loc[df["dataset"] == dataset]
        # Calculate the appropriate bin range
        max_tii = current_df["tii"].max()
        min_tii = current_df["tii"].min()
        bin_start = min_tii - (min_tii % 1) + 0.5
        bin_end = max_tii - (max_tii % 1) + 1.5

        # Plot histogram with binwidth=1 and adjusted binrange
        sns.histplot(
            data=current_df, x="tii", ax=ax, binwidth=1, binrange=(bin_start, bin_end)
        )

        axes[row, col].set_title(dataset + " (n = " + str(len(current_df)) + ")")
        # axes[row, col].set_xlabel("")
        # axes[row, col].set_ylabel("")

    # plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


results_csv = snakemake.input.random_forest_csv
model_features_csv = snakemake.input.model_features_csv
combined_csv = snakemake.input.combined_csv_path
forest_plot_folder = snakemake.params.forest_plot_folder


print("Start plotting TIIs.")
plot_filepath = os.path.join(forest_plot_folder, "tii.pdf")
plot_tiis(combined_csv, plot_filepath)
print("Done plotting TIIs.")

print("Start plotting random forest results.")
random_forest_plot_filepath = os.path.join(
    forest_plot_folder, "random_forest_results.pdf"
)
plot_random_forest_results(results_csv, random_forest_plot_filepath)
model_features_plot_filepath = os.path.join(
    forest_plot_folder, "random_forest_model_features.pdf"
)
plot_random_forest_model_features(model_features_csv, model_features_plot_filepath)
print("Done plotting random forest results.")
