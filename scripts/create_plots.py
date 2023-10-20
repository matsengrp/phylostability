import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from ete3 import Tree
import numpy as np
import re
import pickle
import math
import itertools


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


def aggregate_and_filter_by_likelihood(taxon_edge_csv_list, p, hard_threshold=3):
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
        if len(filtered_df) > hard_threshold:
            filtered_df = filtered_df.nlargest(hard_threshold, "likelihood")
        # append to df for all taxa
        dfs.append(filtered_df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={0: "seq_id"})
    if "Unnamed: 0.1" in df.columns:
        df.set_index("Unnamed: 0.1", inplace=True)
    elif "Unnamed: 0" in df.columns:
        df.set_index("Unnamed: 0", inplace=True)
    return df


def get_ml_dist(ml_dist_file):
    """
    Read ml_dist from input filename and return df with rownames=colnames=seq_ids.
    """
    ml_distances = pd.read_table(
        ml_dist_file,
        skiprows=[0],
        header=None,
        delim_whitespace=True,
        index_col=0,
    )
    ml_distances.columns = ml_distances.index
    return ml_distances


def get_best_reattached_tree(seq_id, all_taxon_edge_df, data_folder):
    """
    Return best reattached tree with seq_id (best means highest likelihood)
    of reattachment edge as given in all_taxon_edge_df.
    """
    filtered_df = all_taxon_edge_df.loc[all_taxon_edge_df["seq_id"] == seq_id]
    if isinstance(filtered_df.index[0], str):
        best_edge_id = filtered_df["likelihood"].idxmin()
    else:
        best_edge_id = filtered_df.loc[filtered_df["likelihood"].idxmin()][0]
    if not isinstance(best_edge_id, str):
        best_edge_id = filtered_df["likelihood"].idxmax()
    best_edge_id = best_edge_id.split("_")[-1]
    tree_filepath = (
        data_folder
        + "reduced_alignments/"
        + seq_id
        + "/reduced_alignment.fasta_add_at_edge_"
        + str(best_edge_id)
        + ".nwk_branch_length.treefile"
    )
    tree = Tree(tree_filepath)
    return tree


def get_best_reattached_tree_distances_to_seq_id(
    seq_id, all_taxon_edge_df, data_folder
):
    """ "
    Return distance matrix for distances in tree with highest likelihood
    among all reattached trees (as per all_taxon_edge_df).
    """
    tree = get_best_reattached_tree(seq_id, all_taxon_edge_df, data_folder)
    leaves = tree.get_leaf_names()
    # Compute distance for each leaf to seq_id leaf
    distances = {}
    for i, leaf in enumerate(leaves):
        if leaf != seq_id:
            distances[leaf] = tree.get_distance(leaf, seq_id)
    return distances


def get_closest_msa_sequences(seq_id, ml_dist_file, p):
    """
    Returns list of names of p closest sequences in MSA to seq_id.
    """
    ml_distances = get_ml_dist(ml_dist_file)
    top_p_rows = ml_distances.nlargest(p, seq_id)
    row_names = top_p_rows.index.tolist()
    return row_names


def get_seq_dists_to_seq_id(seq_id, ml_dist_file, no_seqs=None):
    """
    Returns dict of no_seqs closest distances in MSA to seq_id,
    containing names as keys and distances as values.
    """
    ml_distances = get_ml_dist(ml_dist_file)
    d = {}
    for seq in ml_distances.index:
        if seq != seq_id:
            d[seq] = ml_distances[seq_id][seq]
    # only look at closest no_seqs sequences to seq_id
    if no_seqs != None:
        top_items = sorted(d.items(), key=lambda x: x[1], reverse=False)[:no_seqs]
        top_dict = dict(top_items)
        return top_dict
    return d


def tree_likeness_at_reattachment(
    sorted_taxon_tii_list, all_taxon_edge_df, data_folder, ml_dist_file, plot_filepath
):
    ml_distances = get_ml_dist(ml_dist_file)
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
        avg_tree_dist = 0
        avg_ml_dist = 0
        for leaf1_name in cluster1:
            leaf1 = reduced_tree.search_nodes(name=leaf1_name)[0]
            for leaf2_name in cluster2:
                leaf2 = reduced_tree.search_nodes(name=leaf2_name)[0]
                avg_tree_dist += leaf1.get_distance(leaf2)
                avg_ml_dist += ml_distances[leaf1_name][leaf2_name]
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


def seq_distance_distribution_closest_seq(
    sorted_taxon_tii_list, ml_dist_file, summary_plot_filepath, separate_plots_filename
):
    """
    For each seq_id, find closest sequence in MSA -> closest_sequence.
    Plot difference of MSA distances of all sequences to seq_if and MSA distance of all
    sequences to closest_sequence.
    """
    df = []
    for seq_id, tii in sorted_taxon_tii_list:
        closest_sequence = get_closest_msa_sequences(seq_id, ml_dist_file, 1)[0]
        dist_to_seq = get_seq_dists_to_seq_id(seq_id, ml_dist_file)
        dist_to_closest_seq = get_seq_dists_to_seq_id(closest_sequence, ml_dist_file)
        for i in dist_to_seq:
            if i != closest_sequence:
                df.append(
                    [
                        seq_id + " " + str(tii),
                        closest_sequence,
                        dist_to_closest_seq[i] - dist_to_seq[i],
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
    ml_dist_file,
    data_folder,
    reduced_tree_files,
    plot_filepath,
):
    """ "
    Plot for every seq_id the MSA sequence distances to all sequences of taxa in
    the closest (topologically) clade to the best reattachment of seq_id with lowest
    bootstrap support.
    """
    ml_distances = get_ml_dist(ml_dist_file)
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
    sorted_taxon_tii_list, all_taxon_edge_df, data_folder, ml_dist_file, plot_filepath
):
    """
    For each seq_id, plot distances inside tree vs sequence distances for all pairs of
    taxa in reattached_tree.
    """
    tree_distances = []
    ml_distances = get_ml_dist(ml_dist_file)
    for seq_id, tii in sorted_taxon_tii_list:
        reattached_tree = get_best_reattached_tree(
            seq_id, all_taxon_edge_df, data_folder
        )
        for leaf1, leaf2 in itertools.combinations(reattached_tree.get_leaves(), 2):
            if leaf1 != leaf2:
                dist = leaf1.get_distance(leaf2)
                tree_distances.append(
                    [
                        seq_id + " " + str(tii),
                        dist,
                        ml_distances[leaf1.name][leaf2.name],
                    ]
                )
    df = pd.DataFrame(
        tree_distances, columns=["seq_id", "tree_distance", "ml_distances"]
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
            x="tree_distance",
            y="ml_distances",
            ax=axes[row, col],
        )
        joint_min = min(
            current_df["tree_distance"].min(), current_df["ml_distances"].min()
        )
        joint_max = max(
            current_df["tree_distance"].max(), current_df["ml_distances"].max()
        )

        axes[row, col].plot(
            [joint_min, joint_max], [joint_min, joint_max], color="red", linestyle="--"
        )
        axes[row, col].set_title(tii)
        axes[row, col].set_xlabel("")
        axes[row, col].set_ylabel("")

    # plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def tree_dist_closest_sequences(
    sorted_taxon_tii_list, ml_dist_file, data_folder, p, plot_filepath
):
    """
    Plot pairwise tree distances (in restricted tree) of p taxa that have minimum
    MSA sequence distance to seq_id.
    """
    df = []
    for seq_id, tii in sorted_taxon_tii_list:
        closest_sequences = get_closest_msa_sequences(seq_id, ml_dist_file, p)
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
                df.append([seq_id + " " + str(tii), ete_dist(node1, node2)])
    df = pd.DataFrame(df, columns=["seq_id", "distance"])
    sns.stripplot(data=df, x="seq_id", y="distance")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def plot_seq_and_tree_dist_diff(
    all_taxon_edge_df, sorted_taxon_tii_list, ml_dist_file, data_folder, plot_filepath
):
    """
    Plot difference between distance of seq_id to other sequences in alignment
    and corresponding distance in reattached tree for f in mldist_file.
    """
    ml_distances = get_ml_dist(ml_dist_file)
    distance_diffs = []
    for seq_id, tii in sorted_taxon_tii_list:
        reattached_distance = get_best_reattached_tree_distances_to_seq_id(
            seq_id, all_taxon_edge_df, data_folder
        )
        q = np.quantile(list(reattached_distance.values()), 0.25)
        for seq in reattached_distance:
            if reattached_distance[seq] < q:
                diff = reattached_distance[seq] - ml_distances[seq_id][seq]
                distance_diffs.append([seq_id + str(tii), diff])

    df = pd.DataFrame(distance_diffs, columns=["seq_id", "distance_diff"])
    sns.stripplot(data=df, x="seq_id", y="distance_diff")
    plt.axhline(0, color="red")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def plot_distance_reattachment_sibling(
    all_taxon_edge_df, sorted_taxon_tii_list, ml_dist_file, data_folder, plot_filepath
):
    """
    If S is the subtree that is sibling of reattached sequence, we plots ratio of
    average distance of sequences in S and sequences in S's sibling S' in reduced
    tree to average distance of reattached sequences and sequences in S'.
    """
    ml_distances = get_ml_dist(ml_dist_file)
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

        # TODO: Now we again get nothing for high TII values. This started after introducin conditional_traverse()
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

    for i, taxon in enumerate(best_taxon_edge_df["seq_id"].cat.categories):
        subset = best_taxon_edge_df[best_taxon_edge_df["seq_id"] == taxon]
        plt.scatter(
            [i] * subset.shape[0],
            subset["reattachment_branch_neighbor_distances"],
            c=subset["likelihood"],
            cmap="viridis",
            edgecolors="black",
            linewidth=0.5,
        )

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

all_taxon_edge_df = aggregate_and_filter_by_likelihood(taxon_edge_df_csv, 0.02, 4)
# all_taxon_edge_df = aggregate_taxon_edge_dfs(taxon_edge_df_csv)
print("Done reading data.")


plot_filepath = os.path.join(plots_folder, "tree_vs_sequence_dist_reattached_tree.pdf")
tree_vs_sequence_dist_reattached_tree(
    sorted_taxon_tii_list, all_taxon_edge_df, data_folder, mldist_file, plot_filepath
)


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
p = 3
tree_dist_closest_sequences(
    sorted_taxon_tii_list, mldist_file, data_folder, p, tree_dist_closest_seq_filepath
)
print("Done plotting tree distance between sequences closest to reattachment sequence.")

print("Start plotting sequence and tree distance differences.")
distance_diff_filepath = os.path.join(plots_folder, "distance_diff.pdf")
plot_seq_and_tree_dist_diff(
    all_taxon_edge_df,
    sorted_taxon_tii_list,
    mldist_file,
    data_folder,
    distance_diff_filepath,
)
print("Done plotting sequence and tree distance differences.")


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


# plot branch length distance of reattachment locations vs TII, hue = log_likelihood
# difference
print("Start plotting reattachment distances.")
reattachment_distances_path = os.path.join(
    plots_folder, "dist_of_likely_reattachments.pdf"
)
dist_of_likely_reattachments(
    sorted_taxon_tii_list,
    all_taxon_edge_df,
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
    all_taxon_edge_df,
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

# swarmplot sequence distances from mldist files for each taxon, sort by TII
print("Start plotting MSA sequence distances.")
seq_distance_filepath = os.path.join(plots_folder, "seq_distance_vs_tii.pdf")
seq_distance_swarmplot(
    mldist_file,
    sorted_taxon_tii_list,
    seq_distance_filepath,
)
print("Done plotting MSA sequence distances.")

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
    all_taxon_edge_df, sorted_taxon_tii_list, taxon_height_plot_filepath
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
