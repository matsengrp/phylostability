import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from ete3 import Tree
import numpy as np
import re


taxon_df_csv = snakemake.input.taxon_df_csv
taxon_edge_df_csv = snakemake.input.taxon_edge_df_csv
reduced_tree_files = snakemake.input.reduced_trees
mldist_file = snakemake.input.mldist_file
plots_folder = snakemake.params.plots_folder
reattachment_distance_csv = snakemake.input.reattachment_distance_csv
reattachment_distance_topological_csv = (
    snakemake.input.reattachment_distance_topological_csv
)

taxon_df = pd.read_csv(taxon_df_csv, index_col=0)
taxon_df.index.name = "taxon_name"

taxon_tii_list = [
    (taxon_name, tii) for taxon_name, tii in zip(taxon_df.index, taxon_df["tii"])
]
sorted_taxon_tii_list = sorted(taxon_tii_list, key=lambda x: x[1])


def aggregate_taxon_edge_dfs(csv_list):
    dfs = []
    for csv_file in csv_list:
        taxon_df = pd.read_csv(csv_file)
        dfs.append(taxon_df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={0: "seq_id"})
    return df


def aggregate_and_filter_by_likelihood(taxon_edge_csv_list, p):
    """
    Reads and aggregates taxon_edge_csv_list dataframes for all taxa, while
    also filtering out by likelihood.
    p is a value between 0 and 1, so that only taxon reattachments whose trees have
    likelihood greater than max_likelihood - p * (max_likelihood - min_likelihood)
    are added to the aggregated dataframe.
    """
    dfs = []
    for csv_file in taxon_edge_csv_list:
        taxon_df = pd.read_csv(csv_file)
        # filter by likelihood
        min_likelihood = taxon_df["likelihood"].min()
        max_likelihood = taxon_df["likelihood"].max()
        threshold = max_likelihood - p * (max_likelihood - min_likelihood)
        filtered_df = taxon_df[taxon_df["likelihood"] >= threshold]
        if len(filtered_df) > 3:
            filtered_df = filtered_df.nlargest(3, "likelihood")
        # append to df for all taxa
        dfs.append(filtered_df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={0: "seq_id"})
    return df


def edpl_vs_tii_scatterplot(taxon_df, filepath):
    sns.scatterplot(data=taxon_df, x="tii", y="edpl")
    plt.savefig(filepath)
    plt.clf()


def likelihood_swarmplots(sorted_taxon_tii_list, all_taxon_edge_df, filepath):
    """
    For each taxon, plot the log likelihood of all optimised reattachments as swarmplot,
    sorted according to increasing TII
    """
    all_taxon_edge_df["seq_id"] = pd.Categorical(
        all_taxon_edge_df["seq_id"],
        categories=[
            pair[0] for pair in sorted(sorted_taxon_tii_list, key=lambda x: x[1])
        ],
        ordered=True,
    )
    all_taxon_edge_df = all_taxon_edge_df.sort_values("seq_id")
    plt.figure(figsize=(10, 6))  # Adjust figure size if needed
    sns.stripplot(data=all_taxon_edge_df, x="seq_id", y="likelihood")
    # Add a horizontal line at the maximum likelihood value
    max_likelihood = all_taxon_edge_df["likelihood"].max()
    plt.axhline(y=max_likelihood, color="r")

    # Set labels and title
    plt.xlabel("taxa (sorted by TII)")
    plt.ylabel("Log likelihood")
    plt.title("stripplot of log likelihood vs. taxa sorted by TII")

    plt.xticks(
        range(len(sorted_taxon_tii_list)),
        [
            str(pair[0]) + " " + str(pair[1])
            for pair in sorted(sorted_taxon_tii_list, key=lambda x: x[1])
        ],
    )
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(filepath)
    plt.clf()


def seq_distance_swarmplot(distance_filepath, sorted_taxon_tii_list, plot_filepath):
    """
    For each taxon, plot the sequence distance (from iqtree .mldist file) as swarmplot,
    sorted according to increasing TII
    """
    distances = pd.read_table(
        distance_filepath, skiprows=[0], header=None, delim_whitespace=True, index_col=0
    )
    np.fill_diagonal(distances.values, np.nan)

    # Add seq_id as a column
    distances["seq_id"] = distances.index

    # Reshape the DataFrame into long format
    df_long = pd.melt(
        distances, id_vars=["seq_id"], var_name="variable", value_name="value"
    )

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.stripplot(data=df_long, x="seq_id", y="value")

    # Set labels and title
    plt.xlabel("taxa (sorted by TII)")
    plt.ylabel("distances")
    plt.title("sequence distances vs. taxa sorted by TII")

    # Set x-axis ticks and labels
    plt.xticks(
        range(len(sorted_taxon_tii_list)),
        [
            str(pair[0]) + " " + str(pair[1])
            for pair in sorted(sorted_taxon_tii_list, key=lambda x: x[1])
        ],
        rotation=90,
    )
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def find_reattachment_edge(branchlength_str, seq_id):
    pattern = rf"\['(([\w/.]+,)+[\w/.]+)'"
    matches = re.findall(pattern, branchlength_str)
    matches = [m[0] for m in matches]
    seq_id_clusters = [m for m in matches if seq_id in m]
    if len(seq_id_clusters) > 0:
        new_seq_id_clusters = []
        for cluster in seq_id_clusters:
            cluster_list = cluster.split(",")
            if seq_id in cluster_list:
                cluster_list.remove(seq_id)
            new_seq_id_clusters.append(",".join(cluster_list))
        return min(new_seq_id_clusters, key=lambda s: s.count(","))
    else:
        # we are looking for biggest cluster because seq_id must be in cherry
        # if we couldn't find it above -- is this the only exception?
        return max(matches, key=lambda s: s.count(","))


def get_reattachment_distances(df, reattachment_distance_csv, seq_id):
    """
    Return dataframe with distances between reattachment locations that are represented
    in df for taxon seq_id.
    """
    filtered_df = df[df["seq_id"] == seq_id]
    reattachment_distances_file = [
        csv for csv in reattachment_distance_csv if seq_id in csv
    ][0]

    all_taxa = set(df["seq_id"].unique())

    reattachment_edges = []
    # find identifiers (str of taxon sets) for reattachment edge
    for branchlengths in filtered_df["branchlengths"]:
        reattachment_edges.append(find_reattachment_edge(branchlengths, seq_id))
    reattachment_distances = pd.read_csv(reattachment_distances_file, index_col=0)
    column_names = {}
    for s in reattachment_distances.columns[:-1]:
        column_names[s] = set(s.split(","))
    replace = {}  # save edge identifying strings that need replacement
    for edge_node in reattachment_edges:
        # if cluster of edge_node is not in columns of reattachment_distance,
        # the complement of that cluster will be in there
        edge_node_set = set(edge_node.split(","))
        if edge_node_set not in column_names.values():
            complement_nodes = all_taxa - edge_node_set
            complement_nodes = complement_nodes - set([seq_id])
            complement_node_str = [
                s for s in column_names if column_names[s] == complement_nodes
            ][0]
            replace[edge_node] = complement_node_str
    # make all replacements in reattachmend_edges
    for edge_node in replace:
        reattachment_edges.remove(edge_node)
        reattachment_edges.append(replace[edge_node])

    # update index of reattachment_distancess
    col_list = reattachment_distances.columns.to_list()
    col_list.remove("likelihoods")
    reattachment_distances = reattachment_distances.set_index(pd.Index(col_list))

    # create distance matrix of reattachment positions that are present in input df
    # and again add likelihoods in last column
    filtered_reattachments = reattachment_distances.loc[
        reattachment_edges, reattachment_edges
    ]
    filtered_reattachments["likelihoods"] = reattachment_distances.loc[
        reattachment_edges, "likelihoods"
    ]
    return filtered_reattachments


def dist_of_likely_reattachments(
    sorted_taxon_tii_list, all_taxon_edge_df, reattachment_distance_csv, filepath
):
    """
    Compute ratio of minimum to maximum distance between most likely reattachment locations,
    which we assume are the only ones present in all_taxon_edge_df.
    We plot those values for each taxon, sorted according to increasing TII.
    """
    pairwise_df = []
    # create DataFrame containing min_dist/max_dist ratio as well as
    # max difference between all good likelihood values
    for i, (seq_id, tii) in enumerate(sorted_taxon_tii_list):
        reattachment_distances = get_reattachment_distances(
            all_taxon_edge_df, reattachment_distance_csv, seq_id
        )
        max_likelihood_reattachment = reattachment_distances.loc[
            reattachment_distances["likelihoods"].idxmax()
        ].name
        # create new df with pairwise distances + likelihood difference
        for i in range(len(reattachment_distances)):
            for j in range(i + 1, len(reattachment_distances)):
                best_reattachment = False
                if (
                    reattachment_distances.columns[i] == max_likelihood_reattachment
                ) or (reattachment_distances.columns[j] == max_likelihood_reattachment):
                    best_reattachment = True
                ll_diff = abs(
                    reattachment_distances["likelihoods"][i]
                    - reattachment_distances["likelihoods"][j]
                )
                pairwise_df.append(
                    [
                        seq_id + " " + str(tii),
                        reattachment_distances.columns[i],
                        reattachment_distances.columns[j],
                        reattachment_distances.iloc[i, j],
                        ll_diff,
                        best_reattachment,
                    ]
                )
    pairwise_df = pd.DataFrame(
        pairwise_df,
        columns=[
            "seq_id",
            "edge1",
            "edge2",
            "distance",
            "ll_diff",
            "best_reattachment",
        ],
    )
    # Filter the DataFrame for each marker type

    df_marker_o = pairwise_df[pairwise_df["best_reattachment"] == False]
    df_marker_X = pairwise_df[pairwise_df["best_reattachment"] == True]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    if len(pairwise_df) == 0:
        print("All taxa have unique reattachment locations.")
        plt.savefig(filepath)  # save empty plot to not break snakemake output
        return 0

    # Plot marker 'o'
    sns.stripplot(
        data=df_marker_o,
        x="seq_id",
        y="distance",
        hue="ll_diff",
        palette="viridis",
        alpha=0.7,
        marker="o",
        jitter=True,
        size=7,
    )

    # Plot marker 'X'
    sns.stripplot(
        data=df_marker_X,
        x="seq_id",
        y="distance",
        hue="ll_diff",
        palette="viridis",
        alpha=0.7,
        marker="X",
        jitter=True,
        size=9,
    )

    # Add a colorbar
    norm = plt.Normalize(pairwise_df["ll_diff"].min(), pairwise_df["ll_diff"].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label("ll_diff")

    # Other plot customizations
    plt.legend([], [], frameon=False)
    plt.ylabel("distance between reattachment locations")
    plt.title("Distance between optimal reattachment locations")
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save or display the plot
    plt.savefig(filepath)
    plt.clf()


def bootstrap_swarmplot(reduced_tree_files, sorted_taxon_tii_list, plot_filepath):
    """
    For each taxon, plot the bootstrap support of nodes in the tree inferred on the
    reduced alignment as swarmplot, sorted according to increasing TII
    """
    bootstrap_df = []
    for treefile in reduced_tree_files:
        with open(treefile, "r") as f:
            tree = Tree(f.readlines()[0].strip())
        seq_id = treefile.split("/")[-2]
        tii = [p[1] for p in sorted_taxon_tii_list if p[0] == seq_id][0]
        for node in tree.traverse("postorder"):
            if not node.is_leaf() and not node.is_root():
                bootstrap_df.append([seq_id, node.support, tii])
    bootstrap_df = pd.DataFrame(
        bootstrap_df, columns=["seq_id", "bootstrap_support", "tii"]
    )
    bootstrap_df = bootstrap_df.sort_values("tii")

    plt.figure(figsize=(10, 6))  # Adjust figure size if needed
    sns.stripplot(data=bootstrap_df, x="seq_id", y="bootstrap_support")

    # Set labels and title
    plt.xlabel("taxa (sorted by TII)")
    plt.ylabel("bootstrap support")
    plt.title("stripplot of bootstrap support vs. taxa sorted by TII")

    plt.xticks(
        range(len(sorted_taxon_tii_list)),
        [
            str(pair[0]) + " " + str(pair[1])
            for pair in sorted(sorted_taxon_tii_list, key=lambda x: x[1])
        ],
    )
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def taxon_height_swarmplot(all_taxon_edge_df, sorted_taxon_tii_list, plot_filepath):
    """
    For each taxon, plot the height of the reattachment for all possible reattachment
    edges as a swarmplot vs its TII values
    """
    all_taxon_edge_df["seq_id"] = pd.Categorical(
        all_taxon_edge_df["seq_id"],
        categories=[
            pair[0] for pair in sorted(sorted_taxon_tii_list, key=lambda x: x[1])
        ],
        ordered=True,
    )
    all_taxon_edge_df = all_taxon_edge_df.sort_values("seq_id")
    plt.figure(figsize=(10, 6))  # Adjust figure size if needed
    ax = sns.stripplot(
        data=all_taxon_edge_df,
        x="seq_id",
        y="taxon_height",
        hue="likelihood",
    )

    # Set labels and title
    plt.xlabel("taxa (sorted by TII)")
    plt.ylabel("reattachment height")
    plt.title("stripplot of reattachment height vs. taxa sorted by TII")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="log likelihood")

    plt.xticks(
        range(len(sorted_taxon_tii_list)),
        [
            str(pair[0]) + " " + str(pair[1])
            for pair in sorted(sorted_taxon_tii_list, key=lambda x: x[1])
        ],
        rotation=90,
    )

    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def reattachment_branch_length_swarmplot(
    all_taxon_edge_df, sorted_taxon_tii_list, plot_filepath
):
    """
    For each taxon, plot the height of the reattachment for all possible reattachment
    edges as a swarmplot vs its TII values
    """
    all_taxon_edge_df["seq_id"] = pd.Categorical(
        all_taxon_edge_df["seq_id"],
        categories=[
            pair[0] for pair in sorted(sorted_taxon_tii_list, key=lambda x: x[1])
        ],
        ordered=True,
    )
    all_taxon_edge_df = all_taxon_edge_df.sort_values("seq_id")
    plt.figure(figsize=(10, 6))  # Adjust figure size if needed
    ax = sns.stripplot(
        data=all_taxon_edge_df,
        x="seq_id",
        y="reattachment_branch_length",
        hue="likelihood",
    )

    # Set labels and title
    plt.xlabel("taxa (sorted by TII)")
    plt.ylabel("reattachment branch length")
    plt.title("stripplot of reattachment branch length vs. taxa sorted by TII")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="log likelihood")

    plt.xticks(
        range(len(sorted_taxon_tii_list)),
        [
            str(pair[0]) + " " + str(pair[1])
            for pair in sorted(sorted_taxon_tii_list, key=lambda x: x[1])
        ],
        rotation=90,
    )

    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


all_taxon_edge_df = aggregate_and_filter_by_likelihood(taxon_edge_df_csv, 0.05)
# all_taxon_edge_df = aggregate_taxon_edge_dfs(taxon_edge_df_csv)

# plot branch length distance of reattachment locations vs TII, hue = log_likelihood
# difference
reattachment_distances_path = os.path.join(
    plots_folder, "dist_of_likely_reattachments.pdf"
)
dist_of_likely_reattachments(
    sorted_taxon_tii_list,
    all_taxon_edge_df,
    reattachment_distance_csv,
    reattachment_distances_path,
)

# plot topological distance of reattachment locations vs TII, hue = log_likelihood
# difference
reattachment_topological_distances_path = os.path.join(
    plots_folder, "topological_dist_of_likely_reattachments.pdf"
)
dist_of_likely_reattachments(
    sorted_taxon_tii_list,
    all_taxon_edge_df,
    reattachment_distance_topological_csv,
    reattachment_topological_distances_path,
)

# plot edpl vs TII for each taxon
edpl_filepath = os.path.join(plots_folder, "edpl_vs_tii.pdf")
edpl_vs_tii_scatterplot(taxon_df, edpl_filepath)

# swarmplot likelihoods of reattached trees for each taxon, sort by TII
ll_filepath = os.path.join(plots_folder, "likelihood_swarmplots.pdf")
likelihood_swarmplots(sorted_taxon_tii_list, all_taxon_edge_df, ll_filepath)

# swarmplot sequence distances from mldist files for each taxon, sort by TII
seq_distance_filepath = os.path.join(plots_folder, "seq_distance_vs_tii.pdf")
seq_distance_swarmplot(
    mldist_file,
    sorted_taxon_tii_list,
    seq_distance_filepath,
)

# swarmplot bootstrap support reduced tree for each taxon, sort by TII
bootstrap_plot_filepath = os.path.join(plots_folder, "bootstrap_vs_tii.pdf")
bootstrap_swarmplot(reduced_tree_files, sorted_taxon_tii_list, bootstrap_plot_filepath)

taxon_height_plot_filepath = os.path.join(plots_folder, "taxon_height_vs_tii.pdf")
taxon_height_swarmplot(
    all_taxon_edge_df, sorted_taxon_tii_list, taxon_height_plot_filepath
)
reattachment_branch_length_plot_filepath = os.path.join(
    plots_folder, "reattachment_branch_length_vs_tii.pdf"
)

reattachment_branch_length_swarmplot(
    all_taxon_edge_df, sorted_taxon_tii_list, reattachment_branch_length_plot_filepath
)
