import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from ete3 import Tree
import numpy as np
import re
import pickle


def ete_dist(node1, node2, topology_only=False):
    # if one of the nodes is a leaf and child of the other one, we need to add one
    # to their distance because get_distance() returns number of nodes between
    # given nodes. E.g. if node1 and node2 are connected by edge, this would be
    # distance 0, but it should be 1
    add_to_dist = 0
    if node2 in node1.get_ancestors():
        leaf = node1.get_leaves()[0]
        if node1 == leaf and topology_only == True:
            add_to_dist = 1
        return (
            node2.get_distance(leaf, topology_only=topology_only)
            - node1.get_distance(leaf, topology_only=topology_only)
            + add_to_dist
        )
    else:
        leaf = node2.get_leaves()[0]
        if node2 == leaf and topology_only == True and node1 in node2.get_ancestors():
            add_to_dist = 1
        return (
            node1.get_distance(leaf, topology_only=topology_only)
            - node2.get_distance(leaf, topology_only=topology_only)
            + add_to_dist
        )

def aggregate_taxon_edge_dfs(csv_list):
    """
    Aggregate all dataframes in csv_list into one dataframe for plotting.
    """
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

    plt.xlabel("taxa (sorted by TII)")
    plt.ylabel("Log likelihood")
    plt.title("stripplot of log likelihood vs. taxa sorted by TII")
    plt.xticks(
        range(len(sorted_taxon_tii_list)),
        [
            str(pair[0]) + " " + str(pair[1])
            for pair in sorted(sorted_taxon_tii_list, key=lambda x: x[1])
        ],
        rotation=90,
    )
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

    plt.xlabel("taxa (sorted by TII)")
    plt.ylabel("distances")
    plt.title("sequence distances vs. taxa sorted by TII")
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
    """
    Return ID for edge on which seq_id is attached in branchlength_str.
    This ID is a string containing of the leaf names whose cluster corresponds
    to one of the end nodes of the edge.
    """
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
        # return smallest cluster containing seq_id, as this must be the one of the
        # edge on which seq_id is attached
        return min(new_seq_id_clusters, key=lambda s: s.count(","))
    else:
        # if we haven't found a cluster with sequence_id, then seq_id must be in
        # the complement of all clusters. In this case we want to return the biggest
        # cluster
        return max(matches, key=lambda s: s.count(","))


def get_reattachment_distances(df, reattachment_distance_csv, seq_id):
    """
    Return dataframe with distances between reattachment locations that are represented
    in df for taxon seq_id.
    This assumes tha the reattachment locations in df are a subset of those in
    reattachment_distance_csv.
    """
    filtered_df = df[df["seq_id"] == seq_id]
    reattachment_distances_file = [
        csv for csv in reattachment_distance_csv if seq_id in csv
    ][0]

    all_taxa = set(df["seq_id"].unique())

    reattachment_edges = []
    # find IDs (str of taxon sets) of reattachment edges
    for branchlengths in filtered_df["branchlengths"]:
        reattachment_edges.append(find_reattachment_edge(branchlengths, seq_id))
    reattachment_distances = pd.read_csv(reattachment_distances_file, index_col=0)
    column_names = {}
    for s in reattachment_distances.columns[:-1]:
        column_names[s] = set(s.split(","))

    # because of rooting, the edge IDs in reattachment_distances might be the
    # complement of the edge IDs we get from find_reattachment_edge()
    # we save those in replace (dict) and change the identifier in
    # reattachment_edges
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
    Plot distance between all pairs of reattachment locations in all_taxon_edge_df.
    We plot these distances for each taxon, sorted according to increasing TII, and
    colour the datapooints by log likelihood difference.
    """
    pairwise_df = []
    # create DataFrame containing all distances and likelihood
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
    # Filter the DataFrame for each marker type -- cross for distances to best
    # reattachment position
    df_marker_o = pairwise_df[pairwise_df["best_reattachment"] == False]
    df_marker_X = pairwise_df[pairwise_df["best_reattachment"] == True]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(pairwise_df) == 0:
        print("All taxa have unique reattachment locations.")
        plt.savefig(filepath)  # save empty plot to not break snakemake output
        return 0

    # Plot marker 'o' (only if more than two datapoints)
    if len(df_marker_o) > 0:
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

    # Add colorbar
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
    plt.savefig(filepath)
    plt.clf()


def get_bootstrap_and_bts_scores(
    reduced_tree_files,
    full_tree_file,
    sorted_taxon_tii_list,
    test=False,
):
    """
    Returns three DataFrames:
    (i) branch_score_df containing bts values for all edges in tree in full_tree_file
    (ii) bootstrap_df: bootstrap support by seq_id for reduced trees (without seq_id)
    (iii) full_tree_bootstrap_df: bootstrap support for tree in full_tree_file
    If test==True, we save the pickled bts_score DataFrame in "test_data/bts_df.p"
    to then be able to use it for testing.
    """
    bootstrap_df = []

    with open(full_tree_file, "r") as f:
        full_tree = Tree(f.readlines()[0].strip())

    num_leaves = len(full_tree)

    # initialise branch score dict with sets of leaves representing edges of
    # full_tree
    branch_scores = {}
    all_taxa = full_tree.get_leaf_names()
    full_tree_bootstrap = {}
    for node in full_tree.traverse("postorder"):
        if not node.is_leaf() and not node.is_root():
            sorted_leaves = sorted(node.get_leaf_names())
            leaf_str = ",".join(sorted_leaves)
            full_tree_bootstrap[leaf_str] = node.support
            # ignore pendant edges
            if len(sorted_leaves) < len(all_taxa) - 1:
                s = 0
                for child in node.children:
                    if child.is_leaf():
                        s += 1
                # add 0 for branch score
                branch_scores[",".join(sorted_leaves)] = 0
    full_tree_bootstrap_df = pd.DataFrame(
        full_tree_bootstrap, index=["bootstrap_support"]
    ).transpose()

    for treefile in reduced_tree_files:
        with open(treefile, "r") as f:
            tree = Tree(f.readlines()[0].strip())
        seq_id = treefile.split("/")[-2]
        tii = [p[1] for p in sorted_taxon_tii_list if p[0] == seq_id][0]

        for node in tree.traverse("postorder"):
            if not node.is_leaf() and not node.is_root():
                # add bootstrap support values for seq_id
                bootstrap_df.append([seq_id, node.support, tii])
                # one edge in the full tree could correspond to two edges in the
                # reduced tree (leaf_str and leaf_str_extended below), if the pruned
                # taxon is attached on that edge. We hence need to add one for each of
                # those, if they are in branch_scores.
                edge_found = False
                leaf_list = node.get_leaf_names()
                leaf_str = ",".join(sorted(leaf_list))
                if leaf_str in branch_scores:
                    branch_scores[leaf_str] += 1
                    edge_found = True
                leaf_str_extended = ",".join(sorted(leaf_list + [seq_id]))
                if leaf_str_extended in branch_scores:
                    branch_scores[leaf_str_extended] += 1
                if edge_found:
                    continue
                # edge ID might be complement of leaf set
                # this could happen if rooting of tree is different to that of full_tree
                complement_leaves = list(
                    set(all_taxa) - set(node.get_leaf_names()) - set(seq_id)
                )
                # ignore node if it represents leaf
                if len(complement_leaves) == 1:
                    continue
                leaf_str = ",".join(complement_leaves)
                if leaf_str in branch_scores:
                    branch_scores[leaf_str] += 1
                leaf_str_extended = ",".join(complement_leaves + [seq_id])
                if leaf_str_extended in branch_scores:
                    branch_scores[leaf_str_extended] += 1
    # normalise BTS
    # Divide by num_leaves - 2 if edge es incident to cherry, as it cannot be present in
    # either of the two trees where one of those cherry leaves is pruned
    for branch_score in branch_scores:
        if branch_score.count(",") == 1 or branch_score.count(",") == num_leaves - 3:
            branch_scores[branch_score] *= 100 / (num_leaves - 2)  # normalise bts
        else:
            branch_scores[branch_score] *= 100 / num_leaves
        branch_scores[branch_score] = int(branch_scores[branch_score])
    branch_scores_df = pd.DataFrame(branch_scores, index=["bts"]).transpose()

    bootstrap_df = pd.DataFrame(
        bootstrap_df, columns=["seq_id", "bootstrap_support", "tii"]
    )
    bootstrap_df = bootstrap_df.sort_values("tii")

    if test == True:
        with open("test_data/bts_df.p", "wb") as f:
            pickle.dump(branch_scores_df, file=f)
    return branch_scores_df, bootstrap_df, full_tree_bootstrap_df


def bootstrap_and_bts_plot(
    reduced_tree_files,
    full_tree_file,
    sorted_taxon_tii_list,
    bts_plot_filepath,
    bootstrap_plot_filepath,
    bts_vs_bootstrap_path,
):
    (
        branch_scores_df,
        bootstrap_df,
        full_tree_bootstrap_df,
    ) = get_bootstrap_and_bts_scores(
        reduced_tree_files, full_tree_file, sorted_taxon_tii_list, test=True
    )

    # plot BTS values
    sns.swarmplot(data=branch_scores_df, x="bts")
    plt.title("BTS")
    plt.tight_layout()
    plt.savefig(bts_plot_filepath)
    plt.clf()

    # plot BTS vs bootstrap values
    # sort both dataframes so we plot corresponding values correctly
    branch_scores_df = branch_scores_df.sort_values(
        by=list(branch_scores_df.columns)
    ).reset_index(drop=True)
    full_tree_bootstrap_df = full_tree_bootstrap_df.sort_values(
        by=list(full_tree_bootstrap_df.columns)
    ).reset_index(drop=True)
    merged_df = pd.concat([branch_scores_df, full_tree_bootstrap_df], axis=1)
    sns.scatterplot(data=merged_df, x="bootstrap_support", y="bts")
    plt.title("BTS vs Bootstrap Support")
    plt.xlabel("bootstrap support in full tree")
    plt.ylabel("branch taxon score (bts)")
    plt.tight_layout()
    plt.savefig(bts_vs_bootstrap_path)
    plt.clf()

    # Set labels and title
    plt.xlabel("taxa (sorted by TII)")
    plt.ylabel("bootstrap support")
    plt.title("stripplot of bootstrap support vs. taxa sorted by TII")

    # plot bootstrap support of reduced alignments vs tii
    plt.figure(figsize=(10, 6))  # Adjust figure size if needed
    sns.stripplot(data=bootstrap_df, x="seq_id", y="bootstrap_support")
    plt.xticks(
        range(len(sorted_taxon_tii_list)),
        [
            str(pair[0]) + " " + str(pair[1])
            for pair in sorted(sorted_taxon_tii_list, key=lambda x: x[1])
        ],
        rotation=90,
    )

    plt.tight_layout()
    plt.savefig(bootstrap_plot_filepath)
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

def seq_distance_differences_swarmplot(distance_filepath, ete_filepath, sorted_taxon_tii_list, plot_filepath):
    """
    For each taxon, plot the ratio of the sequence distance (from iqtree .mldist file) to the
    topological distance as swarmplot, sorted according to increasing TII
    """
    ml_distances = pd.read_table(
        distance_filepath, skiprows=[0], header=None, delim_whitespace=True, index_col=0
    )
    np.fill_diagonal(ml_distances.values, np.nan)
    ml_distances = pd.DataFrame(ml_distances).rename(columns={i+1:x for i, x in enumerate(ml_distances.index)})

    with open(ete_filepath, "r") as f:
       whole_tree = Tree(f.readlines()[0].strip())
    tp_distances = pd.DataFrame({
            seq_id: [ete_dist(whole_tree & seq_id, whole_tree & other_seq, topology_only=True) \
                     for other_seq in ml_distances.index] \
            for seq_id in ml_distances.index}, \
    ).transpose().rename(columns={i:x for i, x in enumerate(ml_distances.index)})

    distances = pd.DataFrame([ml_distances[seq_id].divide(tp_distances[seq_id]) for seq_id in tp_distances.columns])

    # Add seq_id as a column
    distances["seq_id"] = ml_distances.index

    # Reshape the DataFrame into long format
    df_long = pd.melt(
        distances, id_vars=["seq_id"], var_name="variable", value_name="value"
    )

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.stripplot(data=df_long, x="seq_id", y="value")

    # Set labels and title
    plt.xlabel("taxa (sorted by TII)")
    plt.ylabel("ratio of computed and topological distances")
    plt.title("sequence distances ratios vs. taxa sorted by TII")

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



taxon_df_csv = snakemake.input.taxon_df_csv
taxon_edge_df_csv = snakemake.input.taxon_edge_df_csv
reduced_tree_files = snakemake.input.reduced_trees
full_tree_file = snakemake.input.full_tree
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
bts_plot_filepath = os.path.join(plots_folder, "bts_scores.pdf")
bts_vs_bootstrap_path = os.path.join(plots_folder, "bts_vs_bootstrap.pdf")
bootstrap_and_bts_plot(
    reduced_tree_files,
    full_tree_file,
    sorted_taxon_tii_list,
    bts_plot_filepath,
    bootstrap_plot_filepath,
    bts_vs_bootstrap_path,
)

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

seq_dist_difference_plot_filepath = os.path.join(
    plots_folder, "sequence_distance_differences.pdf"
)
seq_distance_differences_swarmplot(mldist_file, full_tree_file, sorted_taxon_tii_list, seq_dist_difference_plot_filepath)
