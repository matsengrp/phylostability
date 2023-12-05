import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import ast

from utils import *


plt.rcParams.update({"font.size": 12})  # Adjust this value as needed
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12


def bootstrap_and_bts_plot(
    bootstrap_csv,
    plot_filepath,
):
    """
    Plot BTS vs bootstrap support values of full tree
    """
    # plot BTS vs bootstrap values
    merged_df = pd.read_csv(bootstrap_csv)

    # Compute the 2D histogram
    hist, xedges, yedges = np.histogram2d(
        merged_df["bootstrap_support"], merged_df["bts"], bins=50
    )

    # Use the histogram values to assign each data point a density value
    xidx = np.clip(
        np.digitize(merged_df["bootstrap_support"], xedges), 0, hist.shape[0] - 1
    )
    yidx = np.clip(np.digitize(merged_df["bts"], yedges), 0, hist.shape[1] - 1)
    merged_df["density"] = hist[xidx, yidx]

    # Create the scatter plot with colors based on density and a logarithmic scale
    plt.scatter(
        merged_df["bootstrap_support"],
        merged_df["bts"],
        c=merged_df["density"],
        cmap="RdBu_r",
    )

    cb = plt.colorbar(label="density")

    # Set other plot properties
    plt.title("BTS vs Bootstrap Support")
    plt.xlabel("bootstrap support in full tree")
    plt.ylabel("branch taxon score (bts)")
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def df_column_swarmplot(csv, col_name, plot_filepath):
    """
    For each taxon, plot the value in column col_name vs TII values
    """
    df = pd.read_csv(csv)

    # Check if the column contains string representations of lists
    if isinstance(df.iloc[0][col_name], str):
        df[col_name] = df[col_name].apply(ast.literal_eval)
        df = df.explode(col_name)

    # Convert the column to numeric, to handle any inadvertent string or object types
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    df_sorted = df.sort_values(by="tii")

    plt.figure(figsize=(10, 6))
    ax = sns.stripplot(data=df_sorted, x="seq_id", y=col_name)

    # Set labels and title
    plt.xlabel("taxa (sorted by TII)")
    plt.ylabel(col_name)
    plt.title(col_name + " vs. taxa sorted by TII")

    plt.xticks(rotation=90)

    # Shading every second TII value
    unique_tii = sorted(df_sorted["tii"].unique())
    for i in range(1, len(unique_tii), 2):
        tii_vals = df_sorted["seq_id"][df_sorted["tii"] == unique_tii[i]].unique()
        for val in tii_vals:
            idx = list(df_sorted["seq_id"].unique()).index(val)
            ax.axvspan(
                idx - 0.5,
                idx + 0.5,
                facecolor="lightgrey",
                edgecolor=None,
                alpha=0.5,
            )

def plot_random_forest_results(results_csv, plot_filepath):
    df = pd.read_csv(results_csv, index_col=0)
    df_sorted = df.sort_values(by="actual").melt("actual", var_name="model", value_name="predicted_value")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_sorted, x="actual", y="predicted_value", hue="model")
    plt.title("results of random forest regression")
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def plot_random_forest_model_features(model_features_csv, plot_filepath):
    df = pd.read_csv(
        model_features_csv, names=["feature_name", "untuned", "tuned"], skiprows=1, header=0
    ).melt("feature_name", var_name="model_type", value_name="importance")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="feature_name", y="importance", hue="model_type")
    plt.title("feature importance for random forest regression")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def plot_random_forest_classifier_results(results_csv, plot_filepath):
    df = pd.read_csv(results_csv, index_col=0)
    df_sorted = df.sort_values(by="tii value")
    for col in ["actual", "untuned_model_predicted", "predicted"]:
        df_sorted[col] = ["unstable" if x else "stable" for x in df_sorted[col]]
    df_sorted = df_sorted.melt("tii value", var_name="classifier", value_name="predicted_unstable")
    df_sorted = df_sorted[df_sorted["predicted_unstable"] != "stable"]
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_sorted, x="tii value", hue="classifier")
    plt.title("results of random forest classifier")
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


csv = snakemake.input.csv
bootstrap_csv = snakemake.input.bootstrap_csv

plots_folder = snakemake.params.plots_folder

if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

print("Start reading, aggregating, and filtering data.")
taxon_df = pd.read_csv(csv, index_col=0)
taxon_df.index.name = "taxon_name"

taxon_tii_list = [
    (seq_id.split(" ")[0], int(seq_id.split(" ")[1]))
    for seq_id in taxon_df["seq_id"].unique()
]
sorted_taxon_tii_list = sorted(taxon_tii_list, key=lambda x: x[1])
print("Done reading data.")


print("Start plotting reattachment distances.")
plot_filepath = os.path.join(plots_folder, "dist_of_likely_reattachments.pdf")
df_column_swarmplot(csv, "reattachment_distances", plot_filepath)
print("Done plotting reattachment distances.")


print("Start plotting reattachment distance to low support node.")
plot_filepath = os.path.join(
    plots_folder, "reattachment_distance_to_low_support_node.pdf"
)
df_column_swarmplot(csv, "dist_reattachment_low_bootstrap_node", plot_filepath)
print("Done plotting reattachment distance to low support node.")


print("Start plotting NJ TII.")
plot_filepath = os.path.join(plots_folder, "NJ_TII.pdf")
df_column_swarmplot(csv, "nj_tii", plot_filepath)
print("Done plotting NJ TII.")


print("Start plotting order of distances to seq_id in tree vs MSA.")
plot_filepath = os.path.join(plots_folder, "order_of_distances_to_seq_id.pdf")
df_column_swarmplot(csv, "order_diff", plot_filepath)
print("Done plotting order of distances to seq_id in tree vs MSA.")


print("Start plotting sequence and tree distance differences.")
plot_filepath = os.path.join(plots_folder, "seq_and_tree_dist_ratio.pdf")
df_column_swarmplot(csv, "seq_and_tree_dist_ratio", plot_filepath)
print("Done plotting sequence and tree distance differences.")


# swarmplot bootstrap support reduced tree for each taxon, sort by TII
print("Start plotting bootstrap and bts.")
plot_filepath = os.path.join(plots_folder, "bts_vs_bootstrap.pdf")
bootstrap_and_bts_plot(
    bootstrap_csv,
    plot_filepath,
)
print("Done plotting bootstrap and bts.")


# Swarmplots of statistics we can get straight out of csv file
print("Start plotting likelihoods of reattached trees.")
ll_filepath = os.path.join(plots_folder, "likelihood_swarmplots.pdf")
df_column_swarmplot(csv, "likelihood", ll_filepath)
print("Done plotting likelihoods of reattached trees.")

print("Start plotting LWR of reattached trees.")
lwr_filepath = os.path.join(plots_folder, "likelihood_weight_ratio.pdf")
df_column_swarmplot(csv, "like_weight_ratio", lwr_filepath)
print("Done plotting LWR of reattached trees.")

print("Start plotting reattachment heights.")
taxon_height_plot_filepath = os.path.join(plots_folder, "taxon_height_vs_tii.pdf")
df_column_swarmplot(csv, "taxon_height", taxon_height_plot_filepath)
print("Done plotting reattachment heights.")

print("Start plotting reattachment branch length.")
reattachment_branch_length_plot_filepath = os.path.join(
    plots_folder, "reattachment_branch_length_vs_tii.pdf"
)
df_column_swarmplot(
    csv,
    "reattachment_branch_length",
    reattachment_branch_length_plot_filepath,
)
print("Done plotting reattachment branch length.")


print("Start plotting pendant branch length.")
pendant_branch_length_plot_filepath = os.path.join(
    plots_folder, "pendant_branch_length_vs_tii.pdf"
)
df_column_swarmplot(
    csv,
    "pendant_branch_length",
    pendant_branch_length_plot_filepath,
)
print("Done plotting reattachment branch length.")

print(
    "Start plotting difference in distances of seq_id and its closest sequence to all other sequences."
)
plot_filepath = os.path.join(plots_folder, "seq_distance_ratios_closest_seq.pdf")
df_column_swarmplot(csv, "seq_distance_ratios_closest_seq", plot_filepath)
print(
    "Start plotting difference in distances of seq_id and its closest sequence to all other sequences."
)

print(
    "Start plotting ratio of distances of sibling cluster of reattachment to its nearest clade to seq_id distance to nearest clade."
)
plot_filepath = os.path.join(plots_folder, "dist_diff_reattachment_sibling.pdf")
df_column_swarmplot(csv, "dist_diff_reattachment_sibling", plot_filepath)
print(
    "Done plotting ratio of distances of sibling cluster of reattachment to its nearest clade to seq_id distance to nearest clade."
)

results_csv = snakemake.input.random_forest_regression_csv
model_features_csv = snakemake.input.model_features_csv
print("Start plotting random forest regression results.")
random_forest_plot_filepath = os.path.join(plots_folder, "random_forest_results.pdf")
plot_random_forest_results(results_csv, random_forest_plot_filepath)
model_features_plot_filepath = os.path.join(
    plots_folder, "random_forest_model_features.pdf"
)
plot_random_forest_model_features(model_features_csv, model_features_plot_filepath)
print("Done plotting random forest regression results.")

results_csv = snakemake.input.random_forest_classifier_csv
model_features_csv = snakemake.input.discrete_model_features_csv
print("Start plotting random forest classifier results.")
random_forest_plot_filepath = os.path.join(plots_folder, "random_forest_classifier_results.pdf")
plot_random_forest_classifier_results(results_csv, random_forest_plot_filepath)
model_features_plot_filepath = os.path.join(
    plots_folder, "random_forest_discrete_model_features.pdf"
)
plot_random_forest_model_features(model_features_csv, model_features_plot_filepath)
print("Done plotting random forest classifier results.")
