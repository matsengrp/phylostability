import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

# Font sizes for figures
plt.rcParams.update({"font.size": 12})
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16

# Colour for plots
dark2 = mpl.colormaps["Dark2"]


def bootstrap_and_bts_plot(
    df,
    plot_filepath,
):
    """
    Plot BTS vs bootstrap support values of full tree
    """
    # Create the scatter plot with colors based on density and a logarithmic scale
    df["density"] = df.groupby(["bts", "bootstrap"])["bts"].transform("count")
    sns.lineplot(
    data=df,
    x="bts",
    y="bootstrap",
    # hue="dataset"
    # bins=100,  # Adjust the number of bins for both x and y axes as needed
    # cmap="Blues",  # Choose a colormap that fits your preference
    # cbar=True,  # Show the color bar indicating counts
)

    # Set other plot properties
    plt.title("")
    plt.xlim([-1, 101])
    plt.ylim([-1, 101])
    plt.xlabel("bootstrap support")
    plt.ylabel("branch taxon score")
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def combine_dfs(csvs, subdirs):
    df_list = []
    for subdir in subdirs:
        csv = [f for f in csvs if subdir in f][0]
        temp_df = pd.read_csv(csv, index_col=0)
        temp_df["dataset"] = subdir.split("/")[-1]
        df_list.append(temp_df)
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


bootstrap_csvs = snakemake.input.bootstrap_csvs
subdirs = snakemake.params.subdirs
aggregated_bootstrap_csv = snakemake.output.aggregated_bootstrap_csv
plot_filepath = snakemake.output.plot_filepath

df = combine_dfs(bootstrap_csvs, subdirs)
df.to_csv(aggregated_bootstrap_csv)
bootstrap_and_bts_plot(df, plot_filepath)
