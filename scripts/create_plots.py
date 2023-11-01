import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import re
import pickle
import math
import itertools

from utils import *


def reattachment_distance_to_low_support_node(
    sorted_taxon_tii_list, all_taxon_edge_df, data_folder, bootstrap_threshold=1
):
    """
    Plot (topological) distance of reattachment position in best_reattached_tree to
    nearest low bootstrap node for each seq_id.
    """
    df = []
    for seq_id, tii in sorted_taxon_tii_list:
        best_reattached_tree = get_best_reattached_tree(
            seq_id, all_taxon_edge_df, data_folder
        )
        reduced_tree_file = (
            data_folder
            + "reduced_alignments/"
            + seq_id
            + "/reduced_alignment.fasta.treefile"
        )
        reduced_tree = Tree(reduced_tree_file)
        all_bootstraps = [
            node.support
            for node in reduced_tree.traverse()
            if not node.is_root() and not node.is_leaf()
        ]
        q = np.quantile(all_bootstraps, bootstrap_threshold)
        # parent of seq_id is reattachment_node
        reattachment_node = best_reattached_tree.search_nodes(name=seq_id)[0].up
        reattachment_child = [
            node for node in reattachment_node.children if node.name != seq_id
        ][0]
        cluster = reattachment_child.get_leaf_names()
        node_in_reduced_tree = reduced_tree.get_common_ancestor(cluster)
        # find closest node with low bootstrap support
        min_dist_found = float("inf")
        for node in [node for node in reduced_tree.traverse() if not node.is_leaf()]:
            # take max of distance of two endpoints of nodes at which we reattach
            if not node.is_root():
                dist = max(
                    [
                        ete_dist(node, node_in_reduced_tree, topology_only=True),
                        ete_dist(node.up, node_in_reduced_tree, topology_only=True),
                    ]
                )
            else:
                dist = ete_dist(node, node_in_reduced_tree, topology_only=True)
            if node.support < q and dist < min_dist_found:
                min_dist_found = dist
        df.append([seq_id + " " + str(tii), min_dist_found])
    df = pd.DataFrame(df, columns=["seq_id", "distance"])
    sns.scatterplot(data=df, x="seq_id", y="distance")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def nj_tii(
    mldist_file, sorted_taxon_tii_list, data_folder, plot_filepath, ratio_plot_filepath
):
    """
    Compute Neighbour Joining TII and compare to ML TII as well as average ratio of NJ
    tree distances to sequence distances for each reduced tree as a measure of how tree
    like the smaller alignment is.
    """

    full_mldist = get_ml_dist(mldist_file)
    full_tree = compute_nj_tree(full_mldist)
    # Convert tree from biopython to ete format

    df = []
    tree_likeness = []
    for seq_id, tii in sorted_taxon_tii_list:
        f = (
            data_folder
            + "reduced_alignments/"
            + seq_id
            + "/reduced_alignment.fasta.mldist"
        )
        reduced_mldist = get_ml_dist(f)
        reduced_tree = compute_nj_tree(reduced_mldist)

        leaves = reduced_tree.get_leaf_names()
        # compute ratio of nj tree distances and sequence distances
        tl = 0
        for leaf1, leaf2 in itertools.combinations(leaves, 2):
            reduced_tree_dist = reduced_tree.get_distance(leaf1, leaf2)
            reduced_ml_dist = reduced_mldist[leaf1][leaf2]
            tl += reduced_tree_dist / reduced_ml_dist
        tree_likeness.append(
            [
                seq_id,
                leaf1,
                leaf2,
                tl
                / (math.factorial(len(leaves)) / math.factorial(len(leaves) - 2) / 2),
            ]
        )

        rf_dist = full_tree.robinson_foulds(reduced_tree, unrooted_trees=True)[0]
        df.append([seq_id + " " + str(tii), rf_dist])
    df = pd.DataFrame(df, columns=["seq_id", "rf_distance"])
    sns.scatterplot(data=df, x="seq_id", y="rf_distance")
    plt.xticks(
        rotation=90,
    )
    plt.title("NJ TII vs ML TII")
    plt.xlabel("seq_id")
    plt.ylabel("NJ TII")
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()

    tree_likeness_df = pd.DataFrame(
        tree_likeness, columns=["seq_id", "leaf1", "leaf2", "ratio"]
    )
    sns.scatterplot(data=tree_likeness_df, x="seq_id", y="ratio")
    plt.xticks(
        rotation=90,
    )
    plt.title("NJ distance : sequence distance ratio")
    plt.xlabel("seq_id")
    plt.ylabel("Ratio")
    plt.tight_layout()
    plt.savefig(ratio_plot_filepath)
    plt.clf()


def order_of_distances_to_seq_id(
    sorted_taxon_tii_list,
    mldist_file,
    all_taxon_edge_df,
    data_folder,
    plot_filepath,
    barplot_filepath,
    histplot_filepath,
):
    """
    Plot difference in tree and sequence distance for sequences of taxon1 and taxon 2
    where distance of taxon1 to seq_id is smaller than taxon2 to seq_id in best reattached
    tree, but greater in terms of sequence distance, for each possible seq_id.
    """
    mldist = get_ml_dist(mldist_file)
    df = []
    hist_counts = []
    for seq_id, tii in sorted_taxon_tii_list:
        best_reattached_tree = get_best_reattached_tree(
            seq_id, all_taxon_edge_df, data_folder
        )
        leaves = [
            leaf for leaf in best_reattached_tree.get_leaf_names() if leaf != seq_id
        ]
        reduced_tree = Tree(
            data_folder
            + "reduced_alignments/"
            + seq_id
            + "/reduced_alignment.fasta.treefile"
        )
        all_bootstraps = [
            node.support
            for node in reduced_tree.traverse()
            if not node.is_leaf() and not node.is_root()
        ]
        q = np.quantile(all_bootstraps, 0.5)
        subset = []
        for leaf1, leaf2 in itertools.combinations(leaves, 2):
            mrca = reduced_tree.get_common_ancestor([leaf1, leaf2])
            if mrca.support >= q or mrca.support == 1.0:
                continue
            # # # Careful: Restricting by leaves being "on the same side" seems biased by rooting
            # # # we only consider leaves on the same side of seq_id in the tree
            # mrca = best_reattached_tree.get_common_ancestor([leaf1, leaf2])
            # p1 = get_nodes_on_path(best_reattached_tree, seq_id, leaf1)
            # p2 = get_nodes_on_path(best_reattached_tree, seq_id, leaf2)
            # if mrca not in p1 or mrca not in p2:
            #     continue

            # only consider leaf1 and leaf2 if their path in reduced tree contains
            # mostly low bootstrap nodes
            low_support = high_support = 0
            p = get_nodes_on_path(reduced_tree, leaf1, leaf2)
            for node in p:
                if node.support < q:
                    low_support += 1
                else:
                    high_support += 1
            if 2 * high_support >= low_support:
                continue
            # add differences between sequence and tree distance to df if order of
            # distances between leaf1 and leaf2 to seq_id are different in tree and
            # msa distance matrix
            tree_dist_leaf1 = best_reattached_tree.get_distance(
                seq_id, leaf1, topology_only=False
            )
            tree_dist_leaf2 = best_reattached_tree.get_distance(
                seq_id, leaf2, topology_only=False
            )
            seq_dist_leaf1 = mldist[seq_id][leaf1]
            seq_dist_leaf2 = mldist[seq_id][leaf2]
            if (
                tree_dist_leaf1 / tree_dist_leaf2 < 1
                and seq_dist_leaf1 / seq_dist_leaf2 > 1
            ):
                difference = (
                    seq_dist_leaf1 / seq_dist_leaf2 - tree_dist_leaf1 / tree_dist_leaf2
                )
                subset.append([seq_id + " " + str(tii), leaf1, leaf2, difference])
            elif (
                tree_dist_leaf2 / tree_dist_leaf1 < 1
                and seq_dist_leaf2 / seq_dist_leaf1 > 1
            ):
                difference = (
                    seq_dist_leaf2 / seq_dist_leaf1 - tree_dist_leaf2 / tree_dist_leaf1
                )
                subset.append([seq_id + " " + str(tii), leaf2, leaf1, difference])
        subset_df = pd.DataFrame(
            subset, columns=["seq_id", "leaf1", "leaf2", "difference"]
        )
        # # Select the 10 rows with highest "difference"
        # sorted_list = sorted(subset, key=lambda x: x[3], reverse=True)
        # subset = sorted_list[:10]

        for other_seq in [s for s, tii in sorted_taxon_tii_list if s != seq_id]:
            count = subset_df["leaf1"].value_counts().get(other_seq, 0) + subset_df[
                "leaf2"
            ].value_counts().get(other_seq, 0)
            hist_counts.append([seq_id + " " + str(tii), other_seq, count])
        df += subset
    df = pd.DataFrame(df, columns=["seq_id", "leaf1", "leaf2", "difference"])

    plt.figure(figsize=(10, 6))
    sns.stripplot(data=df, x="seq_id", y="difference")
    plt.xticks(
        rotation=90,
    )
    plt.title("Difference between sequence distance and tree distance ratio")
    plt.ylabel("Difference in distance ratios")
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()

    # Count the occurrences of each 'seq_id' in df
    counts = df["seq_id"].value_counts().to_frame()
    tii_list = []
    for seq_id in counts.index:
        tii_list.append(float(seq_id.split(" ")[1]))
    counts["tii"] = tii_list
    counts = counts.sort_values(by="tii")
    counts.columns = ["count", "tii"]

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=counts, x=counts.index, y="count")
    plt.xticks(
        rotation=90,
    )
    plt.ylabel("count")
    plt.title("Number of change in order of distances (tree vs msa) to seq_id")
    plt.tight_layout()
    plt.savefig(barplot_filepath)

    # plots number of other_seq in swapped pair for each seq_id separately
    n = len(sorted_taxon_tii_list)
    num_rows = math.ceil(math.sqrt(n))
    num_cols = math.ceil(n / num_rows)

    counts_df = pd.DataFrame(hist_counts, columns=["seq_id", "other_seq", "count"])
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(15, 15), sharex=True, sharey=True
    )
    for index, (seq_id, tii) in enumerate(sorted_taxon_tii_list):
        row = index // num_cols
        col = index % num_cols
        current_df = counts_df.loc[counts_df["seq_id"] == seq_id + " " + str(tii)]
        current_df = current_df.sort_values(by="count", ascending=True)
        sns.barplot(
            data=current_df,
            x="other_seq",
            y="count",
            ax=axes[row, col],
        )
        axes[row, col].set_title(tii)
        axes[row, col].set_xlabel("")
        axes[row, col].set_ylabel("")
        axes[row, col].set_xticklabels("")

    # plt.tight_layout()
    plt.savefig(histplot_filepath)
    plt.clf()


def reattachment_seq_dist_vs_tree_dist(
    sorted_taxon_tii_list, all_taxon_edge_df, mldist_file, data_folder
):
    """
    Plot difference in sequence and tree distance in optimised reattached
    tree between seq_id and all other taxa.
    """
    ml_distances = get_ml_dist(mldist_file)
    df = []
    for seq_id, tii in sorted_taxon_tii_list:
        best_reattached_tree = get_best_reattached_tree(
            seq_id, all_taxon_edge_df, data_folder
        )
        for leaf in best_reattached_tree.get_leaf_names():
            if leaf != seq_id:
                df.append(
                    [
                        seq_id + " " + str(tii),
                        ml_distances[seq_id][leaf]
                        - best_reattached_tree.get_distance(seq_id, leaf),
                    ]
                )
    df = pd.DataFrame(df, columns=["seq_id", "distance_diff"])
    sns.stripplot(data=df, x="seq_id", y="distance_diff")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def plot_seq_and_tree_dist_diff(
    all_taxon_edge_df,
    sorted_taxon_tii_list,
    mldist_file,
    data_folder,
    diff_plot_filepath,
    ratio_plot_filepath,
):
    """
    Plot difference and ratio of distance of seq_id to other sequences in alignment
    and corresponding distance in reattached tree for f in mldist_file.
    """
    ml_distances = get_ml_dist(mldist_file)
    distance_diffs = []
    distance_ratios = []
    for seq_id, tii in sorted_taxon_tii_list:
        tree = get_best_reattached_tree(seq_id, all_taxon_edge_df, data_folder)
        leaves = [leaf for leaf in tree.get_leaf_names() if leaf != seq_id]
        for leaf in leaves:
            ratio = tree.get_distance(seq_id, leaf) / ml_distances[seq_id][leaf]
            distance_ratios.append([seq_id + " " + str(tii), ratio])

        reattached_distance = get_best_reattached_tree_distances_to_seq_id(
            seq_id, all_taxon_edge_df, data_folder
        )
        q = np.quantile(list(reattached_distance.values()), 1)
        for seq in reattached_distance:
            if reattached_distance[seq] < q:
                diff = reattached_distance[seq] - ml_distances[seq_id][seq]
                distance_diffs.append([seq_id + str(tii), diff])

    df = pd.DataFrame(distance_diffs, columns=["seq_id", "distance_diff"])
    sns.stripplot(data=df, x="seq_id", y="distance_diff")
    # plt.axhline(0, color="red")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(diff_plot_filepath)
    plt.clf()

    df = pd.DataFrame(distance_ratios, columns=["seq_id", "distance_diff"])
    sns.stripplot(data=df, x="seq_id", y="distance_diff")
    # plt.axhline(0, color="red")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(ratio_plot_filepath)
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
        csv for csv in reattachment_distance_csv if "/" + seq_id + "/" in csv
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
        plt.clf()
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
    all_taxon_edge_df,
    test=False,
):
    """
    Returns three DataFrames:
    (i) branch_score_df containing bts values for all edges in tree in full_tree_file
    (ii) bootstrap_df: bootstrap support by seq_id for reduced trees (without seq_id)
    (iii) full_tree_bootstrap_df: bootstrap support for tree in full_tree_file
    (iv) local_bootstrap_df: only low bootstrap values for the clade including or below
    the reattachment position for each seq_id
    If test==True, we save the pickled bts_score DataFrame in "test_data/bts_df.p"
    to then be able to use it for testing.
    """
    bootstrap_df = []
    local_bootstrap_df = []

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

        # get local boostrap values
        # find reattachment position with highest likelihood
        row = all_taxon_edge_df.loc[(all_taxon_edge_df["seq_id"] == seq_id)].nlargest(
            1, "likelihood"
        )
        branchlength_str = row["branchlengths"].iloc()[0]
        reattachment_edge = find_reattachment_edge(branchlength_str, seq_id)
        cluster = reattachment_edge.split(",")
        if len(cluster) > 1:
            ancestor = tree.get_common_ancestor(cluster)
        else:
            ancestor = tree.search_nodes(name=cluster[0])[0].up
        path_to_reattachment = ancestor.get_ancestors()
        path_to_reattachment.reverse()
        path_to_reattachment = path_to_reattachment[1:]  # ignore tree root
        path_to_reattachment.append(ancestor)
        # First find the root of the biggest cluster containing seq_id as leaf
        # and having bootstrap support < 90
        cluster_root = None
        for node in path_to_reattachment:
            if not node.is_root() and node.support < 90:
                cluster_root = node
                break
            else:
                continue

        # stop traversing below a node if that node has support < threshold
        def conditional_traverse(node, threshold=90):
            if node.support <= threshold:
                yield node
                for child in node.children:
                    yield from conditional_traverse(child, threshold)

        # traverse subtree and collect bootstrap support
        if cluster_root != None:
            for node in conditional_traverse(cluster_root, threshold=90):
                if not node.is_leaf():
                    local_bootstrap_df.append(
                        [seq_id, node.support, tii, "above_reattachment"]
                    )
        else:
            for node in conditional_traverse(ancestor, threshold=90):
                if not node.is_leaf() and not node.is_root():
                    local_bootstrap_df.append(
                        [seq_id, node.support, tii, "below_reattachment"]
                    )

        # collect bootstrap support and bts for all nodes in tree
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

    local_bootstrap_df = pd.DataFrame(
        local_bootstrap_df, columns=["seq_id", "bootstrap_support", "tii", "location"]
    )
    local_bootstrap_df = local_bootstrap_df.sort_values("tii")

    bootstrap_df = pd.DataFrame(
        bootstrap_df, columns=["seq_id", "bootstrap_support", "tii"]
    )
    bootstrap_df = bootstrap_df.sort_values("tii")

    if test == True:
        with open("test_data/bts_df.p", "wb") as f:
            pickle.dump(branch_scores_df, file=f)
    return branch_scores_df, bootstrap_df, full_tree_bootstrap_df, local_bootstrap_df


def bootstrap_and_bts_plot(
    reduced_tree_files,
    full_tree_file,
    sorted_taxon_tii_list,
    bts_plot_filepath,
    bootstrap_plot_filepath,
    local_bootstrap_plot_filepath,
    bts_vs_bootstrap_path,
    all_taxon_edge_df,
):
    (
        branch_scores_df,
        bootstrap_df,
        full_tree_bootstrap_df,
        local_bootstrap_df,
    ) = get_bootstrap_and_bts_scores(
        reduced_tree_files,
        full_tree_file,
        sorted_taxon_tii_list,
        all_taxon_edge_df,
        test=False,
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
    plt.savefig(bts_vs_bootstrap_path)
    plt.clf()

    # plot for local bootstrap support
    plt.figure(figsize=(10, 6))
    # Ensure dataframe respects the order of sorted_taxon_tii_list for 'seq_id'
    local_bootstrap_df["seq_id"] = pd.Categorical(
        local_bootstrap_df["seq_id"],
        categories=[pair[0] for pair in sorted_taxon_tii_list],
        ordered=True,
    )

    # Now, you can plot directly:
    plt.figure(figsize=(10, 6))
    sns.stripplot(
        data=local_bootstrap_df, x="seq_id", y="bootstrap_support", hue="location"
    )

    # X-axis labels: Use the sorted taxon-TII pairs to label the x-ticks
    plt.xticks(
        range(len(sorted_taxon_tii_list)),
        [str(pair[0]) + " " + str(pair[1]) for pair in sorted_taxon_tii_list],
        rotation=90,
    )

    plt.xlabel("taxa (sorted by TII)")
    plt.ylabel("bootstrap support")
    plt.title("stripplot of bootstrap support vs. taxa sorted by TII")
    plt.tight_layout()
    plt.savefig(local_bootstrap_plot_filepath)
    plt.clf()

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

    # Set labels and title
    plt.xlabel("taxa (sorted by TII)")
    plt.ylabel("bootstrap support")
    plt.title("stripplot of bootstrap support vs. taxa sorted by TII")
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

    plt.figure(figsize=(10, 6))
    # We'll use a scatter plot to enable the use of a colormap
    norm = plt.Normalize(
        all_taxon_edge_df["likelihood"].min(), all_taxon_edge_df["likelihood"].max()
    )
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    for i, taxon in enumerate(all_taxon_edge_df["seq_id"].cat.categories):
        subset = all_taxon_edge_df[all_taxon_edge_df["seq_id"] == taxon]
        plt.scatter(
            [i] * subset.shape[0],
            subset["taxon_height"],
            c=subset["likelihood"],
            cmap="viridis",
            edgecolors="black",
            linewidth=0.5,
        )

    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label("log likelihood")

    # Set labels and title
    plt.xlabel("taxa (sorted by TII)")
    plt.ylabel("reattachment height")
    plt.title("stripplot of reattachment height vs. taxa sorted by TII")

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
    For each taxon, plot the reattachment branch length for all possible reattachment
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

    # We'll use a scatter plot to enable the use of a colormap
    norm = plt.Normalize(
        all_taxon_edge_df["likelihood"].min(), all_taxon_edge_df["likelihood"].max()
    )
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    for i, taxon in enumerate(all_taxon_edge_df["seq_id"].cat.categories):
        subset = all_taxon_edge_df[all_taxon_edge_df["seq_id"] == taxon]
        plt.scatter(
            [i] * subset.shape[0],
            subset["reattachment_branch_length"],
            c=subset["likelihood"],
            cmap="viridis",
            edgecolors="black",
            linewidth=0.5,
        )

    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label("log likelihood")

    # Set labels and title
    plt.xlabel("taxa (sorted by TII)")
    plt.ylabel("reattachment branch length")
    plt.title("stripplot of reattachment branch length vs. taxa sorted by TII")

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
data_folder = snakemake.params.data_folder
reattachment_distance_csv = snakemake.input.reattachment_distance_csv
reattachment_distance_topological_csv = (
    snakemake.input.reattachment_distance_topological_csv
)

print("Start reading, aggregating, and filtering data.")
taxon_df = pd.read_csv(taxon_df_csv, index_col=0)
taxon_df.index.name = "taxon_name"

taxon_tii_list = [
    (taxon_name, tii) for taxon_name, tii in zip(taxon_df.index, taxon_df["tii"])
]
sorted_taxon_tii_list = sorted(taxon_tii_list, key=lambda x: x[1])

filtered_all_taxon_edge_df = aggregate_and_filter_by_likelihood(
    taxon_edge_df_csv, 0.02, 2
)
all_taxon_edge_df = aggregate_taxon_edge_dfs(taxon_edge_df_csv)
print("Done reading data.")


print("Start plotting reattachment distance to low support node.")
plot_filepath = os.path.join(
    plots_folder, "reattachment_distance_to_low_support_node.pdf"
)
reattachment_distance_to_low_support_node(
    sorted_taxon_tii_list, all_taxon_edge_df, data_folder, bootstrap_threshold=0.2
)
print("Done plotting reattachment distance to low support node.")


print("Start plotting NJ TII.")
plot_filepath = os.path.join(plots_folder, "NJ_TII.pdf")
ratio_plot_filepath = os.path.join(plots_folder, "NJ_tree_likeness.pdf")
nj_tii(
    mldist_file, sorted_taxon_tii_list, data_folder, plot_filepath, ratio_plot_filepath
)
print("Done plotting NJ TII.")


print("Start plotting order of distances to seq_id in tree vs MSA.")
plot_filepath = os.path.join(plots_folder, "order_of_distances_to_seq_id.pdf")
barplot_filepath = os.path.join(plots_folder, "order_of_distances_to_seq_id_count.pdf")
histplot_filepath = os.path.join(
    plots_folder, "order_of_distances_to_seq_id_histplots.pdf"
)
order_of_distances_to_seq_id(
    sorted_taxon_tii_list,
    mldist_file,
    all_taxon_edge_df,
    data_folder,
    plot_filepath,
    barplot_filepath,
    histplot_filepath,
)
print("Done plotting order of distances to seq_id in tree vs MSA.")


print("Start plotting tree vs sequence distances to reattached sequence.")
plot_filepath = os.path.join(plots_folder, "reattachment_seq_dist_vs_tree_dist.pdf")
reattachment_seq_dist_vs_tree_dist(
    sorted_taxon_tii_list, all_taxon_edge_df, mldist_file, data_folder
)
print("Done plotting tree vs sequence distances to reattached sequence.")


print("Start plotting sequence and tree distance differences.")
distance_diff_filepath = os.path.join(plots_folder, "seq_and_tree_dist_diff.pdf")
ratio_plot_filepath = os.path.join(plots_folder, "seq_and_tree_dist_ratio.pdf")
plot_seq_and_tree_dist_diff(
    all_taxon_edge_df,
    sorted_taxon_tii_list,
    mldist_file,
    data_folder,
    distance_diff_filepath,
    ratio_plot_filepath,
)
print("Done plotting sequence and tree distance differences.")


# plot branch length distance of reattachment locations vs TII, hue = log_likelihood
# difference
print("Start plotting reattachment distances.")
reattachment_distances_path = os.path.join(
    plots_folder, "dist_of_likely_reattachments.pdf"
)
dist_of_likely_reattachments(
    sorted_taxon_tii_list,
    filtered_all_taxon_edge_df,
    reattachment_distance_csv,
    reattachment_distances_path,
)
print("Done plotting reattachment distances.")

# plot topological distance of reattachment locations vs TII, hue = log_likelihood
# difference
print("Start plotting topological reattachment distances.")
reattachment_topological_distances_path = os.path.join(
    plots_folder, "topological_dist_of_likely_reattachments.pdf"
)
dist_of_likely_reattachments(
    sorted_taxon_tii_list,
    filtered_all_taxon_edge_df,
    reattachment_distance_topological_csv,
    reattachment_topological_distances_path,
)
print("Done plotting topological reattachment distances.")

# # plot edpl vs TII for each taxon
# edpl_filepath = os.path.join(plots_folder, "edpl_vs_tii.pdf")
# edpl_vs_tii_scatterplot(taxon_df, edpl_filepath)

# swarmplot likelihoods of reattached trees for each taxon, sort by TII
print("Start plotting likelihoods of reattached trees.")
ll_filepath = os.path.join(plots_folder, "likelihood_swarmplots.pdf")
likelihood_swarmplots(sorted_taxon_tii_list, all_taxon_edge_df, ll_filepath)
print("Done plotting likelihoods of reattached trees.")


# swarmplot bootstrap support reduced tree for each taxon, sort by TII
print("Start plotting bootstrap and bts.")
bootstrap_plot_filepath = os.path.join(plots_folder, "bootstrap_vs_tii.pdf")
local_bootstrap_plot_filepath = os.path.join(plots_folder, "local_bootstrap_vs_tii.pdf")
bts_plot_filepath = os.path.join(plots_folder, "bts_scores.pdf")
bts_vs_bootstrap_path = os.path.join(plots_folder, "bts_vs_bootstrap.pdf")
bootstrap_and_bts_plot(
    reduced_tree_files,
    full_tree_file,
    sorted_taxon_tii_list,
    bts_plot_filepath,
    bootstrap_plot_filepath,
    local_bootstrap_plot_filepath,
    bts_vs_bootstrap_path,
    all_taxon_edge_df,
)
print("Done plotting bootstrap and bts.")

print("Start plotting reattachment heights.")
taxon_height_plot_filepath = os.path.join(plots_folder, "taxon_height_vs_tii.pdf")
taxon_height_swarmplot(
    filtered_all_taxon_edge_df, sorted_taxon_tii_list, taxon_height_plot_filepath
)
reattachment_branch_length_plot_filepath = os.path.join(
    plots_folder, "reattachment_branch_length_vs_tii.pdf"
)
print("Done plotting reattachment heights.")

print("Start plotting reattachment branch length.")
reattachment_branch_length_swarmplot(
    all_taxon_edge_df, sorted_taxon_tii_list, reattachment_branch_length_plot_filepath
)
print("Done plotting reattachment branch length.")
