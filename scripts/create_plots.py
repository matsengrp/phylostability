import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


taxon_df_csv = snakemake.input.taxon_df_csv
taxon_edge_df_csv = snakemake.input.taxon_edge_df_csv

taxon_df = pd.read_csv(taxon_df_csv, index_col=0)
taxon_df.index.name = "taxon_name"

taxon_tii_list = [
    (taxon_name, tii) for taxon_name, tii in zip(taxon_df.index, taxon_df["tii"])
]
sorted_taxon_tii_list = sorted(taxon_tii_list, key=lambda x: x[1])


def edpl_vs_tii_scatterplot(taxon_df, filepath):
    sns.scatterplot(data=taxon_df, x="tii", y="edpl")
    plt.savefig(filepath)
    plt.clf()


def get_plot_layout(num_plots):
    """
    This function takes as input the number of plots we want in one pdf and returns
    fig,axes for subplots created with matplotlib.pyplot, so that there are in total
    num_plots subplots.
    If num_plots > 10, we get 5 plots per row, otherwise 3.
    """
    if num_plots > 10:
        num_cols = 5
    else:
        num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 8))
    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    # Remove any extra subplots
    for i in range(num_plots, num_rows * num_cols):
        fig.delaxes(axes[i])
    return fig, axes


def likelihood_swarmplots(sorted_taxon_tii_list, taxon_edge_df_csv, filepath):
    # Create figure for likelihood swarmplots
    num_plots = len(sorted_taxon_tii_list)
    fig, axes = get_plot_layout(num_plots)

    # Iterate through CSV files and taxon names
    for i, (taxon_name, tii) in enumerate(sorted_taxon_tii_list):
        # Find the corresponding CSV file
        csv_file = [file for file in taxon_edge_df_csv if taxon_name in file][0]
        df = pd.read_csv(csv_file)
        likelihood_values = df["likelihood"]

        sns.swarmplot(data=likelihood_values, ax=axes[i])
        axes[i].set_title(f"{taxon_name}")
        axes[i].set_xlabel(f"TII = {tii}")
        axes[i].set_ylabel("Likelihood")
        axes[i].set_xticklabels([])
        axes[i].xaxis.set_ticks_position("none")

    plt.tight_layout()
    # Save the plot to pdf
    plt.savefig(filepath)
    plt.clf()


def seq_distance_swarmplot(distance_filepath, sorted_taxon_tii_list, plot_filepath):
    distances = pd.read_table(
        distance_filepath, skiprows=[0], header=None, delim_whitespace=True, index_col=0
    )
    fig, axes = get_plot_layout(len(sorted_taxon_tii_list))
    # iteratively add swarmplot of distances
    for i, (taxon_name, tii) in enumerate(sorted_taxon_tii_list):
        sns.swarmplot(data=distances.loc[taxon_name].to_list(), ax=axes[i])
        axes[i].set_title(f"{taxon_name}")
        axes[i].set_xlabel(f"TII = {tii}")
        axes[i].set_ylabel("Sequence distance to pruned taxon")
        axes[i].set_xticklabels([])
        axes[i].xaxis.set_ticks_position("none")

    plt.tight_layout()
    # Save the plot to pdf
    plt.savefig(plot_filepath)
    plt.clf()


seq_distance_filepath = os.path.join("plots", "seq_distance_vs_tii.pdf")
seq_distance_swarmplot(
    "data/test_input_alignment.fasta.mldist",
    sorted_taxon_tii_list,
    seq_distance_filepath,
)

edpl_filepath = os.path.join("plots", "edpl_vs_tii.pdf")
edpl_vs_tii_scatterplot(taxon_df, edpl_filepath)


ll_filepath = os.path.join("plots", "likelihood_swarmplots.pdf")
likelihood_swarmplots(sorted_taxon_tii_list, taxon_edge_df_csv, ll_filepath)
