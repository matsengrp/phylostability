import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


taxon_df_csv = snakemake.input.taxon_df_csv
taxon_edge_df_csv = snakemake.input.taxon_edge_df_csv

taxon_df = pd.read_csv(taxon_df_csv, index_col=0)
taxon_df.index.name ="taxon_name"

taxon_tii_list = [(taxon_name, tii) for taxon_name, tii in zip(taxon_df.index, taxon_df['tii'])]
sorted_taxon_tii_list = sorted(taxon_tii_list, key=lambda x: x[1])


def edpl_vs_tii_scatterplot(taxon_df, filepath):
    sns.scatterplot(data = taxon_df, x = "tii", y = "edpl")
    plt.savefig(filepath)
    plt.clf()


def likelihood_swarmplots(sorted_taxon_tii_list, taxon_edge_df_csv, filepath):
    # Create figure for likelihood swarmplots
    num_plots = len(sorted_taxon_tii_list)
    if num_plots > 10:
        num_cols = 5
    else:
        num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 8))

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Iterate through CSV files and taxon names
    for i, (taxon_name, tii) in enumerate(sorted_taxon_tii_list):
        # Find the corresponding CSV file
        csv_file = [file for file in taxon_edge_df_csv if taxon_name in file][0]
        df = pd.read_csv(csv_file)
        likelihood_values = df['likelihood']

        sns.swarmplot(data = likelihood_values, ax=axes[i])
        axes[i].set_title(f'{taxon_name}')
        axes[i].set_xlabel(f'TII: {tii}')
        axes[i].set_ylabel('Likelihood')
        axes[i].set_xticklabels([])
        axes[i].xaxis.set_ticks_position('none')

    # Adjust layout
    plt.tight_layout()

    # Remove any extra subplots
    for i in range(num_plots, num_rows * num_cols):
        fig.delaxes(axes[i])
    # Save the plot to a PDF
    plt.savefig(filepath)
    plt.clf()


edpl_filepath = os.path.join('plots', 'edpl_vs_tii.pdf')
edpl_vs_tii_scatterplot(taxon_df, edpl_filepath)


ll_filepath = os.path.join('plots', 'likelihood_swarmplots.pdf')
likelihood_swarmplots(sorted_taxon_tii_list, taxon_edge_df_csv, ll_filepath)