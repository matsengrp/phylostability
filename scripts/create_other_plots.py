from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import itertools

from utils import *


def NJ_vs_best_reattached_tree_sequence_fit(
    sorted_taxon_tii_list, all_taxon_edge_df, data_folder, mldist_file, plot_filepath
):
    """
    Plot difference of NJ tree taxon distances to sequence distance and
    best reattached tree taxon distances to sequence distances to see
    how good a fit the reattached tree is for the sequence distances.
    """

    def get_average_dist_diff(mldist, tree):
        diff = 0
        num_leaves = len(tree)
        for leaf1, leaf2 in itertools.combinations(tree.get_leaf_names(), 2):
            diff += abs(tree.get_distance(leaf1, leaf2) - mldist[leaf1][leaf2])
        diff /= num_leaves * (num_leaves - 1) / 2
        return diff

    df = []
    mldist = get_ml_dist(mldist_file)
    nj_tree = compute_nj_tree(mldist)
    nj_diff = get_average_dist_diff(mldist, nj_tree)
    for seq_id, tii in sorted_taxon_tii_list:
        best_reattached_tree = get_best_reattached_tree(
            seq_id, all_taxon_edge_df, data_folder
        )
        best_reattached_tree_diff = get_average_dist_diff(mldist, best_reattached_tree)
        df.append([seq_id + " " + str(tii), nj_diff - best_reattached_tree_diff])
    df = pd.DataFrame(df, columns=["seq_id", "diff"])
    sns.scatterplot(data=df, x="seq_id", y="diff")
    plt.xticks(
        rotation=90,
    )
    plt.title(
        "Difference in sequence to tree distances for NJ and best reattached tree"
    )
    plt.xlabel("seq_id")
    plt.ylabel("Distance difference")
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def low_bootstrap_cluster_distances_to_seq_id(
    sorted_taxon_tii_list,
    mldist_file,
    all_taxon_edge_df,
    data_folder,
    plot_filepath,
    bootstrap_threshold=1,
):
    """
    For each seq_id, look at all nodes with low bootstrap support and plot difference in
    avg seq_dist to tree_dist ratio for the two clusters of the children of that low
    support node. Tree distances come from best_reattached_tree and sequence distances
    from the mldist file that iqtree outputs for the full dataset.
    """
    mldist = get_ml_dist(mldist_file)
    df = []
    for seq_id, tii in sorted_taxon_tii_list:
        best_reattached_tree = get_best_reattached_tree(
            seq_id, all_taxon_edge_df, data_folder
        )
        # We need to take bootstrap support from reduced tree
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
        for node in reduced_tree.traverse():
            if node.support <= q and not node.is_leaf() and not node.is_root():
                cluster1 = node.children[0].get_leaf_names()
                cluster2 = node.children[1].get_leaf_names()
                cluster1_dist_ratio = cluster2_dist_ratio = 0
                for leaf in cluster1:
                    cluster1_dist_ratio += mldist[seq_id][
                        leaf
                    ] / best_reattached_tree.get_distance(seq_id, leaf)
                cluster1_dist_ratio /= len(cluster1)
                for leaf in cluster2:
                    cluster2_dist_ratio += mldist[seq_id][
                        leaf
                    ] / best_reattached_tree.get_distance(seq_id, leaf)
                cluster2_dist_ratio /= len(cluster2)
                df.append(
                    [
                        seq_id + " " + str(tii),
                        abs(cluster1_dist_ratio - cluster2_dist_ratio),
                    ]
                )
    df = pd.DataFrame(df, columns=["seq_id", "distance_ratios"])
    sns.scatterplot(data=df, x="seq_id", y="distance_ratios")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_filepath)


def low_bootstrap_seq_vs_tree_dist(
    sorted_taxon_tii_list,
    mldist_file,
    data_folder,
    plot_filepath,
    bootstrap_threshold=1,
):
    """
    Plot ratio of average sequence to tree distance for all taxa in the two clusters
    that are children of nodes with low bootstrap support in reduced tree for each seq_id.
    """
    df = []
    mldist = get_ml_dist(mldist_file)
    for seq_id, tii in sorted_taxon_tii_list:
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
        low_bootstrap_nodes = [
            node
            for node in reduced_tree.traverse()
            if not node.is_root() and not node.is_leaf() and node.support < q
        ]
        for node in low_bootstrap_nodes:
            child1_cluster = node.children[0].get_leaf_names()
            child2_cluster = node.children[1].get_leaf_names()
            tree_dist = 0
            ml_dist = 0
            for leaf1, leaf2 in itertools.product(child1_cluster, child2_cluster):
                tree_dist += reduced_tree.get_distance(leaf1, leaf2)
                ml_dist += mldist[leaf1][leaf2]
            df.append(
                [seq_id + " " + str(tii), node.support, ml_dist / tree_dist, ml_dist]
            )
    df = pd.DataFrame(df, columns=["seq_id", "bootstrap", "ratio", "mldist"])
    sns.scatterplot(data=df, x="seq_id", y="ratio")
    plt.xticks(
        rotation=90,
    )
    plt.title("Ratio of avg tree to avg sequence distance for low bootstrap clusters")
    plt.xlabel("seq_id")
    plt.ylabel("Ratio")
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()

    plot_filepath = plot_filepath.split(".")[0] + "_mldist.pdf"
    sns.scatterplot(data=df, x="seq_id", y="mldist")
    plt.xticks(
        rotation=90,
    )
    plt.title("Average sequence distance for low bootstrap clusters")
    plt.xlabel("seq_id")
    plt.ylabel("Sequence distance")
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def seq_distances_full_vs_reduced_tree(
    sorted_taxon_tii_list,
    mldist_file,
    data_folder,
    plot_filepath,
    bootstrap_threshold=1,
):
    """
    Plot difference between sequence distance matrices in full and reduced data sets
    for all taxa.
    If bootstrap_threshold < 1, the bootstrap_threshold-quantile q is used to filter out
    pairs of taxa with high mrca bootstrap support, i.e. only pairs with mrca bootstrap
    support less than q are plotted.
    """
    full_mldist = get_ml_dist(mldist_file)
    df = []
    for seq_id, tii in sorted_taxon_tii_list:
        reduced_mldist_path = (
            data_folder
            + "reduced_alignments/"
            + seq_id
            + "/reduced_alignment.fasta.mldist"
        )
        reduced_mldist = get_ml_dist(reduced_mldist_path)
        reduced_tree_filepath = (
            data_folder
            + "reduced_alignments/"
            + seq_id
            + "/reduced_alignment.fasta.treefile"
        )
        reduced_tree = Tree(reduced_tree_filepath)
        all_bootstraps = [
            node.support
            for node in reduced_tree.traverse()
            if not node.is_leaf() and not node.is_root()
        ]
        q = np.quantile(all_bootstraps, bootstrap_threshold)
        for leaf1, leaf2 in itertools.combinations(reduced_tree.get_leaf_names(), 2):
            mrca = reduced_tree.get_common_ancestor([leaf1, leaf2])
            if mrca.support < q and mrca.support != 1.0:
                df.append(
                    [
                        seq_id + " " + str(tii),
                        leaf1,
                        leaf2,
                        full_mldist[leaf1][leaf2] - reduced_mldist[leaf1][leaf2],
                        tii,
                    ]
                )
    df = pd.DataFrame(df, columns=["seq_id", "leaf1", "leaf2", "dist_diff", "tii"])
    # categorise by whether the entry in dist_diff of df is positive or negative
    df["dist_diff_category"] = pd.cut(
        df["dist_diff"],
        bins=[-float("inf"), 0, float("inf")],
        labels=["Negative", "Positive"],
    )
    df["dist_diff_category"] = np.where(df["dist_diff"] < 0, "negative", "positive")
    count_df = (
        df.groupby(["seq_id", "tii", "dist_diff_category"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    count_df = count_df.sort_values(by="tii")
    count_df = count_df.drop(columns=["tii"])
    count_df = count_df.set_index("seq_id")
    ax = count_df.plot(kind="bar", width=0.8, figsize=(10, 6))
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Negative and Positive dist_diff for Each seq_id")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_filepath)

    # sns.scatterplot(data=df, x="seq_id", y="dist_diff")
    # plt.xticks(
    #     rotation=90,
    # )
    # plt.title("Difference in sequence distance full and reduced alignment")
    # plt.xlabel("seq_id")
    # plt.ylabel("Distance difference")
    # plt.tight_layout()
    # plt.savefig(plot_filepath)
    # plt.clf()


def branch_changes_at_reattachment(
    sorted_taxon_tii_list, all_taxon_edge_df, data_folder, plot_filepath
):
    """
    Plot difference in branch-length distance in reduced tree and optimised reattached
    tree between taxa in clusters (i) below reattachment position (ii) sibling of
    reattachment location.
    """
    df = []
    for seq_id, tii in sorted_taxon_tii_list:
        best_reattached_tree = get_best_reattached_tree(
            seq_id, all_taxon_edge_df, data_folder
        )
        parent = best_reattached_tree.search_nodes(name=seq_id)[0].up
        # get clusters of nodes around reattachment so we can identify corresponding nodes
        # in reduced_tree
        if parent.is_root():
            # TODO: Think about how to deal with the root here!
            continue
        clusternode1 = [node for node in parent.children if node.name != seq_id][0]
        cluster1 = clusternode1.get_leaf_names()
        clusternode2 = [node for node in parent.up.children if node != parent][0]
        cluster2 = clusternode2.get_leaf_names()
        reduced_tree_file = [f for f in reduced_tree_files if "/" + seq_id + "/" in f][
            0
        ]
        reduced_tree = Tree(reduced_tree_file)
        for leaf1_name in cluster1:
            leaf1_reduced = reduced_tree.search_nodes(name=leaf1_name)[0]
            leaf1_reattached = best_reattached_tree.search_nodes(name=leaf1_name)[0]
            for leaf2_name in cluster2:
                leaf2_reduced = reduced_tree.search_nodes(name=leaf2_name)[0]
                leaf2_reattached = best_reattached_tree.search_nodes(name=leaf2_name)[0]
                df.append(
                    [
                        seq_id + " " + str(tii),
                        leaf1_reduced.get_distance(leaf2_reduced)
                        - leaf1_reattached.get_distance(leaf2_reattached),
                    ]
                )
    df = pd.DataFrame(df, columns=["seq_id", "distance_diff"])
    sns.stripplot(data=df, x="seq_id", y="distance_diff")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def tree_likeness_at_reattachment(
    sorted_taxon_tii_list, all_taxon_edge_df, data_folder, mldist_file, plot_filepath
):
    """
    Plot difference in branch-length distance and sequence distance between taxa in clusters
    (i) below reattachment position (ii) sibling of reattachment location.
    """
    ml_distances = get_ml_dist(mldist_file)
    df = []
    for seq_id, tii in sorted_taxon_tii_list:
        best_reattached_tree = get_best_reattached_tree(
            seq_id, all_taxon_edge_df, data_folder
        )
        parent = best_reattached_tree.search_nodes(name=seq_id)[0].up
        # get clusters of nodes around reattachment so we can identify corresponding nodes
        # in reduced_tree
        if parent.is_root():
            # TODO: Think about how to deal with the root here!
            continue
        clusternode1 = [node for node in parent.children if node.name != seq_id][0]
        cluster1 = clusternode1.get_leaf_names()
        clusternode2 = [node for node in parent.up.children if node != parent][0]
        cluster2 = clusternode2.get_leaf_names()
        reduced_tree_file = [f for f in reduced_tree_files if "/" + seq_id + "/" in f][
            0
        ]
        reduced_tree = Tree(reduced_tree_file)
        for leaf1_name in cluster1:
            leaf1 = reduced_tree.search_nodes(name=leaf1_name)[0]
            for leaf2_name in cluster2:
                leaf2 = reduced_tree.search_nodes(name=leaf2_name)[0]
                df.append(
                    [
                        seq_id + " " + str(tii),
                        leaf1.get_distance(leaf2)
                        - ml_distances[leaf1_name][leaf2_name],
                    ]
                )
    df = pd.DataFrame(df, columns=["seq_id", "distance_diff"])
    sns.stripplot(data=df, x="seq_id", y="distance_diff")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def topological_tree_dist_closest_msa_sequence(
    sorted_taxon_tii_list, mldist_file, all_taxon_edge_df, data_folder, plot_filepath
):
    """'
    Plot topological distance of seq_id to taxon with closest sequence distance to seq_id
    in best reattached tree for each seq_id.
    """
    df = []
    for seq_id, tii in sorted_taxon_tii_list:
        tree = get_best_reattached_tree(seq_id, all_taxon_edge_df, data_folder)
        closest_sequences = get_closest_msa_sequences(seq_id, mldist_file, 3)
        all_leaf_dists = {}
        for leaf in tree.get_leaf_names():
            if leaf != seq_id:
                all_leaf_dists[leaf] = tree.get_distance(
                    seq_id, leaf, topology_only=True
                )
        for closest_sequence in closest_sequences:
            closest_leaf_dist = tree.get_distance(
                seq_id, closest_sequence, topology_only=True
            )
            # take ratio of closest_leaf_dist to minimum leaf distance
            closest_leaf_dist /= min(all_leaf_dists.values())
            df.append([seq_id + " " + str(tii), closest_sequence, closest_leaf_dist])
    df = pd.DataFrame(df, columns=["seq_id", "closest_sequence", "tree_dist"])
    sns.scatterplot(data=df, x="seq_id", y="tree_dist")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def msa_distance_closest_topological_dist(
    sorted_taxon_tii_list, mldist_file, all_taxon_edge_df, data_folder, plot_filepath
):
    """'
    Plot topological distance of seq_id to taxon with closest sequence distance to seq_id
    in best reattached tree for each seq_id.
    """
    df = []
    mldist = get_ml_dist(mldist_file)
    for seq_id, tii in sorted_taxon_tii_list:
        tree = get_best_reattached_tree(seq_id, all_taxon_edge_df, data_folder)
        all_leaf_dists = {}
        for leaf in tree.get_leaf_names():
            if leaf != seq_id:
                all_leaf_dists[leaf] = tree.get_distance(
                    seq_id, leaf, topology_only=True
                )
        closest_leaf_name, closest_leaf_dist = min(
            all_leaf_dists.items(), key=lambda x: x[1]
        )
        # take ratio of closest_leaf_dist to minimum leaf distance
        closest_leaf_seq_dist = mldist[seq_id][closest_leaf_name]
        closest_leaf_seq_dist /= mldist[seq_id].max()
        df.append([seq_id + " " + str(tii), closest_leaf_name, closest_leaf_seq_dist])
    df = pd.DataFrame(df, columns=["seq_id", "closest_sequence", "tree_dist"])
    sns.scatterplot(data=df, x="seq_id", y="tree_dist")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def seq_distance_distribution_closest_seq(
    sorted_taxon_tii_list, mldist_file, summary_plot_filepath, separate_plots_filename
):
    """
    For each seq_id, find closest sequence in MSA -> closest_sequence.
    Plot difference of MSA distances of all sequences to seq_if and MSA distance of all
    sequences to closest_sequence.
    """
    df = []
    for seq_id, tii in sorted_taxon_tii_list:
        closest_sequence = get_closest_msa_sequences(seq_id, mldist_file, 1)[0]
        dist_to_seq = get_seq_dists_to_seq_id(seq_id, mldist_file)
        dist_to_closest_seq = get_seq_dists_to_seq_id(closest_sequence, mldist_file)
        for i in dist_to_seq:
            if i != closest_sequence:
                df.append(
                    [
                        seq_id + " " + str(tii),
                        closest_sequence,
                        dist_to_closest_seq[i] / dist_to_seq[i],
                        dist_to_seq[i],
                        dist_to_closest_seq[i],
                    ]
                )
    df = pd.DataFrame(
        df,
        columns=[
            "seq_id",
            "closest_sequence",
            "distance_difference",
            "dist_to_seq",
            "dist_to_closest_seq",
        ],
    )
    sns.stripplot(data=df, x="seq_id", y="distance_difference")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(summary_plot_filepath)
    plt.clf()

    # plots dist_to_seq vs dist_to_closest_seq for each seq_id separately
    n = len(sorted_taxon_tii_list)
    num_rows = math.ceil(math.sqrt(n))
    num_cols = math.ceil(n / num_rows)

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(15, 15), sharex=True, sharey=True
    )
    for index, (seq_id, tii) in enumerate(sorted_taxon_tii_list):
        row = index // num_cols
        col = index % num_cols
        current_df = df.loc[df["seq_id"] == seq_id + " " + str(tii)]
        sns.scatterplot(
            data=current_df,
            x="dist_to_seq",
            y="dist_to_closest_seq",
            ax=axes[row, col],
        )
        joint_min = min(
            current_df["dist_to_seq"].min(), current_df["dist_to_closest_seq"].min()
        )
        joint_max = max(
            current_df["dist_to_seq"].max(), current_df["dist_to_closest_seq"].max()
        )

        axes[row, col].plot(
            [joint_min, joint_max], [joint_min, joint_max], color="red", linestyle="--"
        )
        axes[row, col].set_title(tii)
        axes[row, col].set_xlabel("")
        axes[row, col].set_ylabel("")

    # plt.tight_layout()
    plt.savefig(separate_plots_filename)
    plt.clf()


def seq_distances_to_nearest_low_bootstrap_cluster(
    sorted_taxon_tii_list,
    all_taxon_edge_df,
    mldist_file,
    data_folder,
    reduced_tree_files,
    plot_filepath,
):
    """ "
    Plot for every seq_id the MSA sequence distances to all sequences of taxa in
    the closest (topologically) clade to the best reattachment of seq_id with lowest
    bootstrap support.
    """
    ml_distances = get_ml_dist(mldist_file)
    df = []
    for seq_id, tii in sorted_taxon_tii_list:
        reattached_tree = get_best_reattached_tree(
            seq_id, all_taxon_edge_df, data_folder
        )
        reduced_tree_filepath = [
            f for f in reduced_tree_files if "/" + seq_id + "/" in f
        ][0]
        with open(reduced_tree_filepath, "r") as f:
            reduced_tree = Tree(f.readlines()[0].strip())
        reattachment_cluster = [
            node.get_leaf_names()
            for node in reattached_tree.search_nodes(name=seq_id)[0].up.children
            if node.name != seq_id
        ][0]
        reattachment_node = reduced_tree.get_common_ancestor(reattachment_cluster)
        # find closest node (either above or below reattachment) with bootstrap support
        # below threshold
        bootstrap_values = [
            node.support for node in reduced_tree.traverse() if not node.is_leaf()
        ]
        threshold = np.quantile(bootstrap_values, 0.5)
        upper_candidate = lower_candidate = None
        if not reattachment_node.is_leaf():
            for node in reattachment_node.traverse():
                if node.support < threshold:
                    lower_candidate = node
        for node in reattachment_node.get_ancestors():
            if node.support < threshold:
                upper_candidate = node
        if lower_candidate != None and upper_candidate != None:
            closest_candidate = (
                lower_candidate
                if ete_dist(node, lower_candidate, topology_only=True)
                < ete_dist(node, upper_candidate, topology_only=True)
                else upper_candidate
            )
        elif lower_candidate != None:
            closest_candidate = lower_candidate
        elif upper_candidate != None:
            closest_candidate = upper_candidate
        else:
            continue
        closest_candidate_cluster = closest_candidate.get_leaf_names()
        for leaf in closest_candidate_cluster:
            df.append([seq_id + " " + str(tii), ml_distances[leaf][seq_id]])
    df = pd.DataFrame(df, columns=["seq_id", "distance"])
    sns.stripplot(data=df, x="seq_id", y="distance")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def tree_vs_sequence_dist_reattached_tree(
    sorted_taxon_tii_list,
    all_taxon_edge_df,
    data_folder,
    mldist_file,
    plot_filepath,
    mrca_bootstrap_filter=1,
):
    """
    For each seq_id, plot distances inside tree vs sequence distances for all pairs of
    taxa in reduced_tree (i.e. excluding seq_id).
    We only plot pairs of taxon that have mrca with bootstrap support in lower
    mrca_bootstrap_filter-quantile in reduced_tree.
    """
    tree_distances = []
    ml_distances = get_ml_dist(mldist_file)
    for seq_id, tii in sorted_taxon_tii_list:
        reattached_tree = get_best_reattached_tree(
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
            if not node.is_leaf() and not node.is_root()
        ]
        q = np.quantile(all_bootstraps, mrca_bootstrap_filter)
        for leaf1, leaf2 in itertools.combinations(reduced_tree.get_leaf_names(), 2):
            mrca = reduced_tree.get_common_ancestor([leaf1, leaf2])
            if mrca.support < q and not mrca.support == 1.0:
                reattached_dist = reattached_tree.get_distance(leaf1, leaf2)
                reduced_dist = reduced_tree.get_distance(leaf1, leaf2)
                tree_distances.append(
                    [
                        seq_id + " " + str(tii),
                        reattached_dist,
                        reduced_dist,
                        ml_distances[leaf1][leaf2],
                    ]
                )
    df = pd.DataFrame(
        tree_distances,
        columns=[
            "seq_id",
            "reattached_tree_distance",
            "reduced_tree_distance",
            "ml_distances",
        ],
    )
    n = len(sorted_taxon_tii_list)
    num_rows = math.ceil(math.sqrt(n))
    num_cols = math.ceil(n / num_rows)

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(15, 15), sharex=True, sharey=True
    )
    for index, (seq_id, tii) in enumerate(sorted_taxon_tii_list):
        row = index // num_cols
        col = index % num_cols
        current_df = df.loc[df["seq_id"] == seq_id + " " + str(tii)]
        sns.scatterplot(
            data=current_df,
            x="reattached_tree_distance",
            y="ml_distances",
            ax=axes[row, col],
        )
        joint_min = min(
            current_df["reattached_tree_distance"].min(),
            current_df["ml_distances"].min(),
        )
        joint_max = max(
            current_df["reattached_tree_distance"].max(),
            current_df["ml_distances"].max(),
        )

        axes[row, col].plot(
            [joint_min, joint_max], [joint_min, joint_max], color="red", linestyle="--"
        )
        axes[row, col].set_title(seq_id + " " + str(tii))
        axes[row, col].set_xlabel("")
        axes[row, col].set_ylabel("")

    # plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def tree_dist_closest_sequences(
    sorted_taxon_tii_list, mldist_file, data_folder, p, plot_filepath
):
    """
    Plot pairwise tree distances (in restricted tree) of p taxa that have minimum
    MSA sequence distance to seq_id.
    """
    df = []
    for seq_id, tii in sorted_taxon_tii_list:
        closest_sequences = get_closest_msa_sequences(seq_id, mldist_file, p)
        path_to_tree = (
            data_folder
            + "reduced_alignments/"
            + seq_id
            + "/reduced_alignment.fasta.treefile"
        )
        with open(path_to_tree, "r") as f:
            tree = Tree(f.readlines()[0].strip())
        for i in range(len(closest_sequences)):
            leaf1 = closest_sequences[i]
            node1 = tree.search_nodes(name=leaf1)[0]
            for j in range(i + 1, len(closest_sequences)):
                leaf2 = closest_sequences[j]
                node2 = tree.search_nodes(name=leaf2)[0]
                df.append(
                    [
                        seq_id + " " + str(tii),
                        ete_dist(node1, node2, topology_only=True),
                    ]
                )
    df = pd.DataFrame(df, columns=["seq_id", "distance"])
    sns.swarmplot(data=df, x="seq_id", y="distance")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def seq_dist_closest_sequences(sorted_taxon_tii_list, mldist_file, p, plot_filepath):
    """
    Plot pairwise sequence distances of p taxa that have minimum
    MSA sequence distance to seq_id.
    """
    df = []
    for seq_id, tii in sorted_taxon_tii_list:
        closest_sequences = get_closest_msa_sequences(seq_id, mldist_file, p)
        mldist = get_ml_dist(mldist_file)
        for seq in closest_sequences:
            df.append([seq_id + " " + str(tii), seq, mldist[seq_id][seq]])
    df = pd.DataFrame(df, columns=["seq_id", "other seq", "distance"])
    sns.swarmplot(data=df, x="seq_id", y="distance")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def plot_distance_reattachment_sibling(
    all_taxon_edge_df, sorted_taxon_tii_list, mldist_file, data_folder, plot_filepath
):
    """
    If S is the subtree that is sibling of reattached sequence, we plots ratio of
    average distance of sequences in S and sequences in S's sibling S' in reduced
    tree to average distance of reattached sequences and sequences in S'.
    """
    ml_distances = get_ml_dist(mldist_file)
    distances = []
    for seq_id, tii in sorted_taxon_tii_list:
        # TODO: Ideally we look at the distances within the tree rather than the sequence
        # distance(?). This does however require knowing optimised branch lengths in the
        # reattached tree!
        tree = get_best_reattached_tree(seq_id, all_taxon_edge_df, data_folder)
        reattachment_node = tree.search_nodes(name=seq_id)[0].up
        if reattachment_node.is_root():
            # for now we ignore reattachments just under the root
            continue
        sibling = [node for node in reattachment_node.children if node.name != seq_id][
            0
        ]
        sibling_cluster = sibling.get_leaf_names()
        siblings_sibling = [
            node for node in reattachment_node.up.children if node != reattachment_node
        ][0]
        siblings_sibling_cluster = siblings_sibling.get_leaf_names()
        avg_sibling_distance = 0
        avg_new_node_distance = 0
        for leaf1 in siblings_sibling_cluster:
            avg_new_node_distance += ml_distances[leaf1][seq_id]
            for leaf2 in sibling_cluster:
                avg_sibling_distance += ml_distances[leaf1][leaf2]
        avg_sibling_distance /= len(sibling_cluster) * len(siblings_sibling_cluster)
        avg_new_node_distance /= len(siblings_sibling_cluster)
        distances.append(
            [seq_id + " " + str(tii), avg_sibling_distance, "sibling_distance"]
        )
        distances.append(
            [seq_id + " " + str(tii), avg_new_node_distance, "new_node_distance"]
        )
        distances.append(
            [
                seq_id + " " + str(tii),
                abs(avg_new_node_distance - avg_sibling_distance),
                "diff",
            ]
        )

    df = pd.DataFrame(distances, columns=["seq_id", "distances", "class"])
    sns.stripplot(data=df[df["class"] == "diff"], x="seq_id", y="distances")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def edpl_vs_tii_scatterplot(taxon_df, filepath):
    sns.scatterplot(data=taxon_df, x="tii", y="edpl")
    plt.savefig(filepath)
    plt.clf()


def seq_distances(
    distance_filepath,
    sorted_taxon_tii_list,
    all_taxon_edge_df,
    data_folder,
    plot_filepath,
    top5_only=False,
    bootstrap_threshold=1,
):
    """
    For each taxon, plot the sequence distance (from iqtree .mldist file) as swarmplot,
    sorted according to increasing TII.
    We can filter the sequence distances we plot by
    (i) top5_only: only plot distances of 5 leaves that are closeset to reattachment position
    (ii) bootstrap_threshold < 1: only plot distance of leaves where on the path between
    best reattachment and leaf more than half of the nodes have bootstrap support < bootstrap_threshold
    """
    distances = get_ml_dist(distance_filepath)
    np.fill_diagonal(distances.values, np.nan)
    df = []
    plt.figure(figsize=(10, 6))
    if top5_only:
        for seq_id, tii in sorted_taxon_tii_list:
            best_reattached_tree = get_best_reattached_tree(
                seq_id, all_taxon_edge_df, data_folder
            )
            seq_id_node = best_reattached_tree.search_nodes(name=seq_id)[0]
            tree_dists = {
                leaf.name: seq_id_node.get_distance(leaf)
                for leaf in best_reattached_tree.get_leaves()
            }
            top_5_keys = sorted(tree_dists, key=tree_dists.get, reverse=True)[:5]
            for key in top_5_keys:
                df.append([seq_id + " " + str(tii), distances[seq_id][key]])
        df = pd.DataFrame(df, columns=["seq_id", "distance"])
        sns.stripplot(data=df, x="seq_id", y="distance")
    elif bootstrap_threshold < 1:
        for seq_id, tii in sorted_taxon_tii_list:
            reduced_tree_file = (
                data_folder
                + "reduced_alignments/"
                + seq_id
                + "/reduced_alignment.fasta.treefile"
            )
            reduced_tree = Tree(reduced_tree_file)
            leaves = reduced_tree.get_leaf_names()
            all_bootstraps = [
                node.support
                for node in reduced_tree.traverse()
                if not node.is_root() and not node.is_leaf()
            ]
            q = np.quantile(all_bootstraps, bootstrap_threshold)

            best_reattached_tree = get_best_reattached_tree(
                seq_id, all_taxon_edge_df, data_folder
            )
            seq_id_node = best_reattached_tree.search_nodes(name=seq_id)[0]
            # find node above which seq_id got reattached
            below_seq_id = [
                child for child in seq_id_node.up.children if child.name != seq_id
            ][0].get_leaf_names()
            below_seq_id = reduced_tree.get_common_ancestor(below_seq_id)
            for leaf in leaves:
                p = get_nodes_on_path(reduced_tree, below_seq_id, leaf)
                low_support = high_support = 0
                for node in p:
                    if node.support < q:
                        low_support += 1
                    else:
                        high_support += 1
                if high_support >= low_support:
                    continue
                df.append([seq_id + " " + str(tii), leaf, distances[seq_id][leaf]])
        df = pd.DataFrame(df, columns=["seq_id", "leaf", "distances"])
        sns.stripplot(data=df, x="seq_id", y="distances")
    else:
        distances["seq_id"] = distances.index
        df_long = pd.melt(
            distances, id_vars=["seq_id"], var_name="variable", value_name="value"
        )
        sns.stripplot(data=df_long, x="seq_id", y="value")

    plt.xlabel("taxa (sorted by TII)")
    plt.ylabel("distances")
    plt.title("sequence distances vs. taxa sorted by TII")
    plt.xticks(
        #     range(len(sorted_taxon_tii_list)),
        #     [
        #         str(pair[0]) + " " + str(pair[1])
        #         for pair in sorted(sorted_taxon_tii_list, key=lambda x: x[1])
        #     ],
        rotation=90,
    )
    plt.tight_layout()
    if bootstrap_threshold < 1:
        plot_filepath = plot_filepath.split(".")[0]
        plot_filepath += "_bootstrap_threshold_" + str(bootstrap_threshold) + ".pdf"
    elif top5_only:
        plot_filepath = plot_filepath.split(".")[0]
        plot_filepath += "_top5_only.pdf"

    plt.savefig(plot_filepath)
    plt.clf()


def get_sequence_distance_difference(distance_file, taxon1, taxon2, taxon3):
    distances = pd.read_table(
        distance_file, skiprows=[0], header=None, delim_whitespace=True, index_col=0
    )
    distances.columns = distances.index
    d1 = distances.loc[taxon1, taxon2].sum().sum()
    d2 = distances.loc[taxon1, taxon3].sum().sum()
    return min(d1 / (d2 if d2 != 0 else 1.0), d2 / (d1 if d1 != 0 else 1.0))


def reattachment_branch_distance_ratio_plot(
    best_taxon_edge_df, seq_distance_file, sorted_taxon_tii_list, plot_filepath
):
    best_taxon_edge_df["seq_id"] = pd.Categorical(
        best_taxon_edge_df["seq_id"],
        categories=[
            pair[0] for pair in sorted(sorted_taxon_tii_list, key=lambda x: x[1])
        ],
        ordered=True,
    )
    best_taxon_edge_df = all_taxon_edge_df.sort_values("seq_id")
    best_taxon_edge_df["reattachment_branch_neighbor_distances"] = [
        get_sequence_distance_difference(
            seq_distance_file,
            taxon,
            best_taxon_edge_df.loc[
                all_taxon_edge_df.seq_id == taxon, "reattachment_parent"
            ].values,
            best_taxon_edge_df.loc[
                all_taxon_edge_df.seq_id == taxon, "reattachment_child"
            ].values,
        )
        for taxon in best_taxon_edge_df["seq_id"]
    ]

    plt.figure(figsize=(10, 6))  # Adjust figure size if needed

    # We'll use a scatter plot to enable the use of a colormap
    norm = plt.Normalize(
        best_taxon_edge_df["likelihood"].min(), best_taxon_edge_df["likelihood"].max()
    )
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    i = 0
    for taxon, tii in sorted_taxon_tii_list:
        subset = best_taxon_edge_df[best_taxon_edge_df["seq_id"] == taxon]
        plt.scatter(
            [i] * subset.shape[0],
            subset["reattachment_branch_neighbor_distances"],
            c=subset["likelihood"],
            cmap="viridis",
            edgecolors="black",
            linewidth=0.5,
        )
        i += 1

    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label("log likelihood")

    # Set labels and title
    plt.xlabel("taxa (sorted by TII)")
    plt.ylabel("reattachment branch distance ratio")
    plt.title("stripplot of reattachment neighbor distances vs. taxa sorted by TII")

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


def seq_distance_differences_swarmplot(
    distance_filepath, ete_filepath, sorted_taxon_tii_list, plot_filepath
):
    """
    For each taxon, plot the ratio of the sequence distance (from iqtree .mldist file) to the
    topological distance as swarmplot, sorted according to increasing TII
    """
    ml_distances = pd.read_table(
        distance_filepath, skiprows=[0], header=None, delim_whitespace=True, index_col=0
    )
    np.fill_diagonal(ml_distances.values, np.nan)
    ml_distances = pd.DataFrame(ml_distances).rename(
        columns={i + 1: x for i, x in enumerate(ml_distances.index)}
    )

    with open(ete_filepath, "r") as f:
        whole_tree = Tree(f.readlines()[0].strip())
    tp_distances = (
        pd.DataFrame(
            {
                seq_id: [
                    ete_dist(
                        whole_tree & seq_id, whole_tree & other_seq, topology_only=True
                    )
                    for other_seq in ml_distances.index
                ]
                for seq_id in ml_distances.index
            },
        )
        .transpose()
        .rename(columns={i: x for i, x in enumerate(ml_distances.index)})
    )

    distances = pd.DataFrame(
        [
            ml_distances[seq_id].divide(tp_distances[seq_id])
            for seq_id in tp_distances.columns
        ]
    )

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


def mldist_plots(
    mldist_file, sorted_taxon_tii_list, plot_filepath, closest_taxa_plot_filepath
):
    """
    Plot mldist[i][j]/(mldist[j][seq_id]+mldist[i][seq_id]) for every seq_id and all i,j.
    """
    mldist = pd.read_table(
        mldist_file, skiprows=[0], header=None, delim_whitespace=True, index_col=0
    )
    mldist.columns = mldist.index
    df = []
    for seq_id in [l[0] for l in sorted_taxon_tii_list]:
        other_seqs = mldist.columns.to_list()
        other_seqs.remove(seq_id)
        for i in other_seqs:
            for j in other_seqs:
                if i != j:
                    df.append(
                        [
                            seq_id,
                            i,
                            j,
                            mldist[i][j] / (mldist[i][seq_id] + mldist[j][seq_id]),
                        ]
                    )
    df = pd.DataFrame(df, columns=["seq_id", "t1", "t2", "distance_ratio"])

    # Sort the dataframe by 'seq_id'
    sorted_names = [x[0] for x in sorted_taxon_tii_list]
    sorted_tii_values = [x[1] for x in sorted_taxon_tii_list]
    df["seq_id"] = pd.Categorical(df["seq_id"], categories=sorted_names, ordered=True)
    df = df.sort_values("seq_id").reset_index(drop=True)

    sns.stripplot(data=df[df["distance_ratio"] > 1], x="seq_id", y="distance_ratio")
    plt.xlabel("taxa (sorted by TII)")
    plt.ylabel("ratio of tree distance to sum of sequence distance for this taxon")
    plt.title("sequence distances ratios vs. taxa sorted by TII")
    plt.xticks(
        range(len(sorted_names)),
        [
            str(pair[0]) + " " + str(pair[1])
            for pair in sorted(sorted_taxon_tii_list, key=lambda x: x[1])
        ],
        rotation=90,
    )
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()

    # check if seq_id is so closer to i that its previously closest taxon j
    # now gets pushed over to another taxon m
    ratios = {}
    for seq_id in mldist.index:
        i = mldist.loc[seq_id].drop(seq_id).idxmin()
        j = mldist.loc[i].drop([seq_id, i]).idxmin()
        m = mldist.loc[j].drop([seq_id, i, j]).idxmin()
        ratio = mldist.loc[j, m] / ((mldist.loc[seq_id, j] + mldist.loc[i, j]) / 2)
        ratios[seq_id] = ratio

    ratios_series = pd.Series(ratios)
    ratios_series = ratios_series[sorted_names]

    # Plotting
    ratios_series.plot(kind="bar", figsize=(10, 6))
    plt.ylabel("Computed Ratio")
    plt.title("Ratio for each seq_id")
    tick_labels = [
        f"{name} ({tii})" for name, tii in zip(sorted_names, sorted_tii_values)
    ]
    plt.xticks(range(len(tick_labels)), tick_labels, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(closest_taxa_plot_filepath)
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

plots_folder = os.path.join(plots_folder, "other_plots/")
Path(plots_folder).mkdir(parents=True, exist_ok=True)

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


print(
    "Start plotting avg cluster distance difference to seq_id for children of low bootstrap nodes."
)
plot_filepath = os.path.join(
    plots_folder, "low_bootstrap_cluster_distances_to_seq_id.pdf"
)
low_bootstrap_cluster_distances_to_seq_id(
    sorted_taxon_tii_list,
    mldist_file,
    all_taxon_edge_df,
    data_folder,
    plot_filepath,
    bootstrap_threshold=0.2,
)
print(
    "Done plotting avg cluster distance difference to seq_id for children of low bootstrap nodes."
)


print("Start plotting tree fit sequence data of NJ vs reattached tree.")
plot_filepath = os.path.join(
    plots_folder, "NJ_vs_best_reattached_tree_sequence_fit.pdf"
)
NJ_vs_best_reattached_tree_sequence_fit(
    sorted_taxon_tii_list, all_taxon_edge_df, data_folder, mldist_file, plot_filepath
)
print("Start plotting tree fit sequence data of NJ vs reattached tree.")


print(
    "Start plotting ratios of avg tree to avg sequence distance for low bootstrap nodes."
)
plot_filepath = os.path.join(plots_folder, "low_bootstrap_seq_vs_tree_dist.pdf")
low_bootstrap_seq_vs_tree_dist(
    sorted_taxon_tii_list,
    mldist_file,
    data_folder,
    plot_filepath,
    bootstrap_threshold=0.1,
)
print(
    "Start plotting ratios of avg tree to avg sequence distance for low bootstrap nodes."
)


print("Start plotting difference in sequence distances, full vs reduced alignments")
plot_filepath = os.path.join(
    plots_folder, "sequence_distance_diff_full_vs_reduced_tree.pdf"
)
seq_distances_full_vs_reduced_tree(
    sorted_taxon_tii_list,
    mldist_file,
    data_folder,
    plot_filepath,
    bootstrap_threshold=0.05,
)
print("Done plotting difference in sequence distances, full vs reduced alignment.")

print("Start plotting sequence distance to taxon closest in tree.")
plot_filepath = os.path.join(
    plots_folder, "msa_sequence_dist_closest_topological_dist.pdf"
)
msa_distance_closest_topological_dist(
    sorted_taxon_tii_list, mldist_file, all_taxon_edge_df, data_folder, plot_filepath
)
print("Done plotting sequence distance to taxon closest in tree.")


print("Start plotting tree distance to closest MSA sequence.")
plot_filepath = os.path.join(
    plots_folder, "topological_tree_dist_closest_msa_sequence.pdf"
)
topological_tree_dist_closest_msa_sequence(
    sorted_taxon_tii_list, mldist_file, all_taxon_edge_df, data_folder, plot_filepath
)
print("Done plotting tree distance to closest MSA sequence.")


print(
    "Start plotting sequence distance between sequences closest to reattached sequence."
)
plot_filepath = os.path.join(plots_folder + "seq_dist_closest_sequences.pdf")
p = 5
seq_dist_closest_sequences(sorted_taxon_tii_list, mldist_file, p, plot_filepath)
print(
    "Done plotting sequence distance between sequences closest to reattached sequence."
)


print("Start plotting tree vs sequence distances at reattachment.")
plot_filepath = os.path.join(plots_folder, "tree_vs_sequence_dist_reattached_tree.pdf")
tree_vs_sequence_dist_reattached_tree(
    sorted_taxon_tii_list,
    all_taxon_edge_df,
    data_folder,
    mldist_file,
    plot_filepath,
    mrca_bootstrap_filter=0.1,
)
print("Done plotting tree vs sequence distances at reattachment.")


print("Start plotting branch length changes at reattachment.")
plot_filepath = os.path.join(plots_folder, "branch_changes_at_reattachment.pdf")
branch_changes_at_reattachment(
    sorted_taxon_tii_list, all_taxon_edge_df, data_folder, plot_filepath
)
print("Done plotting branch length changes at reattachment.")


print("Start plotting tree-likeness at reattachment.")
plot_filepath = os.path.join(plots_folder, "tree_likeness_at_reattachment.pdf")
tree_likeness_at_reattachment(
    sorted_taxon_tii_list, all_taxon_edge_df, data_folder, mldist_file, plot_filepath
)
print("Done plotting tree-likeness at reattachment.")


print("Start plotting sequence distance to nearest low bootstrap cluster.")
plot_filepath = os.path.join(
    plots_folder, "seq_distances_to_nearest_low_bootstrap_cluster.pdf"
)
seq_distances_to_nearest_low_bootstrap_cluster(
    sorted_taxon_tii_list,
    all_taxon_edge_df,
    mldist_file,
    data_folder,
    reduced_tree_files,
    plot_filepath,
)
print("Done plotting sequence distance to nearest low bootstrap cluster.")


print(
    "Start plotting difference in MSA distances for seq_id:all and closest_seq_to_seq_id:all."
)
summary_plot_filename = os.path.join(
    plots_folder, "seq_distance_distribution_closest_seq.pdf"
)
separate_plots_filename = os.path.join(
    plots_folder, "seq_distance_distribution_closest_seq_separeate_seq_ids.pdf"
)
seq_distance_distribution_closest_seq(
    sorted_taxon_tii_list, mldist_file, summary_plot_filename, separate_plots_filename
)
print(
    "Start plotting difference in MSA distances for seq_id:all and closest_seq_to_seq_id:all."
)


print(
    "Start plotting tree distance between sequences closest to reattachment sequence."
)
tree_dist_closest_seq_filepath = os.path.join(plots_folder, "tree_dist_closest_seq.pdf")
p = 5
tree_dist_closest_sequences(
    sorted_taxon_tii_list, mldist_file, data_folder, p, tree_dist_closest_seq_filepath
)
print("Done plotting tree distance between sequences closest to reattachment sequence.")


print("Start plotting distances between addded sequence and siblings of reattachment.")
distance_reattachment_sibling_filepath = os.path.join(
    plots_folder, "distance_reattachment_sibling.pdf"
)
plot_distance_reattachment_sibling(
    all_taxon_edge_df,
    sorted_taxon_tii_list,
    mldist_file,
    data_folder,
    distance_reattachment_sibling_filepath,
)
print("Done plotting distances between addded sequence and siblings of reattachment.")

print("Start plotting mldists.")
mldist_plot_filepath = os.path.join(plots_folder, "mldist_ratio.pdf")
mldist_closest_taxa_plot_filepath = os.path.join(
    plots_folder, "mldist_closest_taxa_comparisons.pdf"
)
mldist_plots(
    mldist_file,
    sorted_taxon_tii_list,
    mldist_plot_filepath,
    mldist_closest_taxa_plot_filepath,
)
print("Done plotting mldists.")

# swarmplot sequence distances from mldist files for each taxon, sort by TII
print("Start plotting MSA sequence distances.")
seq_distance_filepath = os.path.join(plots_folder, "seq_distance_vs_tii.pdf")
seq_distances(
    mldist_file,
    sorted_taxon_tii_list,
    all_taxon_edge_df,
    data_folder,
    seq_distance_filepath,
    bootstrap_threshold=0.2,
)
print("Done plotting MSA sequence distances.")

print("Start plotting sequence differences.")
seq_dist_difference_plot_filepath = os.path.join(
    plots_folder, "sequence_distance_differences.pdf"
)
seq_distance_differences_swarmplot(
    mldist_file,
    full_tree_file,
    sorted_taxon_tii_list,
    seq_dist_difference_plot_filepath,
)
print("Done plotting sequence differences.")


print("Start plotting ratio of distances for reattachment locations.")
best_reattachment_dist_plot_filepath = os.path.join(
    plots_folder, "reattachment_edge_distance_ratio.pdf"
)
best_taxon_edge_df = aggregate_and_filter_by_likelihood(taxon_edge_df_csv, 1, 1)
reattachment_branch_distance_ratio_plot(
    best_taxon_edge_df,
    mldist_file,
    sorted_taxon_tii_list,
    best_reattachment_dist_plot_filepath,
)
print("Done plotting ratio of distances for reattachment locations.")