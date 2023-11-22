from ete3 import Tree
import pandas as pd
import json
import numpy as np
import itertools

from utils import *

epa_result_files = snakemake.input.epa_results
full_tree_files = snakemake.input.full_tree
restricted_tree_files = snakemake.input.restricted_trees

full_mldist_files = snakemake.input.full_mldist_file
restricted_mldist_files = snakemake.input.restricted_mldist_files

plot_csvs = snakemake.output.plot_csv
random_forest_csvs = snakemake.output.random_forest_csv
bootstrap_csvs = snakemake.output.bootstrap_csv

subdirs = snakemake.params.subdirs


def calculate_taxon_height(input_tree, taxon_name):
    """
    Return distance of reattachment position to nearest leaf (that is not the reattached leaf)
    """
    taxon = input_tree & taxon_name
    taxon_parent = taxon.up
    return min(
        [
            taxon_parent.get_distance(leaf)
            for leaf in input_tree.get_leaves()
            if leaf != taxon
        ]
    )


def get_nj_tiis(full_mldist_file, restricted_mldist_files):
    """
    Get NJ TII by running NJ on the two input matrices and returning
    RF distance between the two inferred trees.
    """
    full_mldist = get_ml_dist(full_mldist_file)
    full_nj_tree = compute_nj_tree(full_mldist)
    tiis = {}
    # maximum possible RF distance:
    normalising_constant = 2 * len(full_nj_tree) - 3
    for file in restricted_mldist_files:
        seq_id = file.split("/")[-2]
        restricted_mldist = get_ml_dist(file)
        restricted_nj_tree = compute_nj_tree(restricted_mldist)
        tiis[seq_id] = (
            full_nj_tree.robinson_foulds(restricted_nj_tree, unrooted_trees=True)[0]
            / normalising_constant
        )
    return tiis


def get_order_of_distances_to_seq_id(
    seq_id,
    mldist_file,
    reattached_tree,
):
    """
    Get difference in tree and sequence distance for sequences of taxon1 and taxon2
    where distance of taxon1 to seq_id is smaller than taxon2 to seq_id in best reattached
    tree, but greater in terms of sequence distance, for each possible seq_id.
    # We filter and currently only look at pairs of taxa with this property whose mrca
    # has bootstrap support in the lowest 10% of bootstap values throughout the tree.
    """
    mldist = get_ml_dist(mldist_file)
    max_mldist = mldist.max().max()
    dist_diff_dict = []

    # find maximum distance between any two leaves in best_reattached_tree
    max_tree_dist = 0
    for leaf1, leaf2 in itertools.combinations(reattached_tree.get_leaves(), 2):
        if leaf1.get_distance(leaf2) > max_tree_dist:
            max_tree_dist = leaf1.get_distance(leaf2)

    leaves = [leaf for leaf in reattached_tree.get_leaf_names() if leaf != seq_id]
    reattachment_node = reattached_tree.search_nodes(name=seq_id)[0].up
    all_bootstraps = [
        node.support
        for node in reattached_tree.traverse()
        if not node.is_leaf() and not node.is_root() and node != reattachment_node
    ]
    # only consider 5 nodes with lowest bootstrap support (or alternatively all nodes
    # with bootstrap support less than 100)
    all_bootstraps.sort()
    q = np.quantile(bootstrap_list, 1)
    for leaf1, leaf2 in itertools.combinations(leaves, 2):
        mrca = reattached_tree.get_common_ancestor([leaf1, leaf2])
        if mrca.support >= q or mrca.support == 1.0:
            continue
        tree_dist_leaf1 = (
            reattached_tree.get_distance(seq_id, leaf1, topology_only=True)
            / max_tree_dist
        )
        tree_dist_leaf2 = (
            reattached_tree.get_distance(seq_id, leaf2, topology_only=True)
            / max_tree_dist
        )
        seq_dist_leaf1 = mldist[seq_id][leaf1] / max_mldist
        seq_dist_leaf2 = mldist[seq_id][leaf2] / max_mldist
        if (
            tree_dist_leaf1 / tree_dist_leaf2 < 1
            and seq_dist_leaf1 / seq_dist_leaf2 > 1
        ):
            difference = (
                seq_dist_leaf1 / seq_dist_leaf2 - tree_dist_leaf1 / tree_dist_leaf2
            )
            dist_diff_dict.append(difference)
        elif (
            tree_dist_leaf2 / tree_dist_leaf1 < 1
            and seq_dist_leaf2 / seq_dist_leaf1 > 1
        ):
            difference = (
                seq_dist_leaf2 / seq_dist_leaf1 - tree_dist_leaf2 / tree_dist_leaf1
            )
            dist_diff_dict.append(difference)
    return dist_diff_dict


def get_reattachment_distances(reduced_tree, reattachment_trees, seq_id):
    """
    Return list of distances between best reattachment locations given by
    """
    if len(reattachment_trees) == 1:
        return [0]
    reattachment_node_list = []

    # maximum reattachment distance could reach is max dist between any two leaves in the
    # smaller tree
    max_reattachment_dist = max(
        [
            leaf1.get_distance(leaf2, topology_only=True)
            for leaf1, leaf2 in itertools.combinations(reduced_tree.get_leaves(), 2)
        ]
    )
    for tree in reattachment_trees:
        # cluster is set of leaves below lower node of reattachment edge
        cluster = tree.search_nodes(name=seq_id)[0].up.get_leaf_names()
        cluster.remove(seq_id)
        if len(cluster) == 1:  # reattachment above leaf
            reattachment_node = tree.search_nodes(name=cluster[0])[0]
        else:
            mrca = reduced_tree.get_common_ancestor(cluster)
            reattachment_node = mrca
            reattachment_node_list.append(reattachment_node)
    reattachment_distances = []
    for node1, node2 in itertools.combinations(reattachment_node_list, 2):
        reattachment_distances.append(
            ete_dist(node1, node2, topology_only=True) / max_reattachment_dist
        )
    return reattachment_distances


def reattachment_distance_to_low_support_node(
    seq_id, reattached_tree, bootstrap_threshold=0.1
):
    """
    Plot (topological) distance of reattachment position in best_reattached_tree to
    nearest low bootstrap node for each seq_id.
    """
    reattachment_node = reattached_tree.search_nodes(name=seq_id)[0].up
    all_bootstraps = [
        node.support
        for node in reattached_tree.traverse()
        if not node.is_root() and not node.is_leaf() and node != reattachment_node
    ]
    q = np.quantile(all_bootstraps, bootstrap_threshold)
    # parent of seq_id is reattachment_node
    min_dist_found = float("inf")
    max_dist_found = 0  # for normalising
    for node in [
        node
        for node in reattached_tree.traverse()
        if not node.is_leaf() and node != reattachment_node
    ]:
        # take max of distance of two endpoints of nodes at which we reattach
        if not node.is_root():
            dist = max(
                [
                    ete_dist(node, reattachment_node, topology_only=True),
                    ete_dist(node.up, reattachment_node, topology_only=True),
                ]
            )
        else:
            dist = ete_dist(node, reattachment_node, topology_only=True)
        if dist > max_dist_found:
            max_dist_found = dist
        if node.support < q and dist < min_dist_found:
            min_dist_found = dist
    return min_dist_found / max_dist_found


def seq_and_tree_dist_diff(
    seq_id,
    mldist_file,
    reattached_tree,
):
    """
    Get difference and ratio of distance between seq_id and other sequences in alignment
    to corresponding distance in best reattached tree.
    """
    ml_distances = get_ml_dist(mldist_file)
    distance_ratios = []
    leaves = [leaf for leaf in reattached_tree.get_leaf_names() if leaf != seq_id]
    # fill distance_ratios
    for leaf in leaves:
        ratio = ml_distances[seq_id][leaf] / reattached_tree.get_distance(seq_id, leaf)
        distance_ratios.append(ratio)
    return distance_ratios


def get_distance_reattachment_sibling(seq_id, mldist_file, reattached_tree):
    """
    If S is the subtree that is sibling of reattached sequence, we compute ratio of
    average distance of sequences in S and sequences in S's sibling S' in reduced
    tree to average distance of reattached sequences and sequences in S'.
    """
    ml_distances = get_ml_dist(mldist_file)
    reattachment_node = reattached_tree.search_nodes(name=seq_id)[0].up
    if reattachment_node.is_root():
        # for now we ignore reattachments just under the root
        return None
    sibling = [node for node in reattachment_node.children if node.name != seq_id][0]
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
    return abs(avg_new_node_distance - avg_sibling_distance)


def seq_distance_distribution_closest_seq(
    seq_id,
    mldist_file,
    reattached_tree,
):
    """
    Find closest sequence in MSA -> closest_sequence.
    Compute ratio of MSA distances of all sequences to seq_id to MSA distance of all
    sequences to closest_sequence.
    """
    ratios = []
    closest_sequence = get_closest_msa_sequences(seq_id, mldist_file, 1)[0]
    dist_to_seq = get_seq_dists_to_seq_id(seq_id, mldist_file)
    dist_to_closest_seq = get_seq_dists_to_seq_id(closest_sequence, mldist_file)
    for seq in dist_to_seq:
        if seq != closest_sequence:
            seq_id_tree_dist = reattached_tree.get_distance(seq_id, seq)
            closest_seq_tree_dist = reattached_tree.get_distance(closest_sequence, seq)
            # we can interpret the following ratios as the msa distance normalised by
            # branch lengths, i.e. as distance per branch length unit
            seq_id_ratio = dist_to_seq[seq] / seq_id_tree_dist
            closest_seq_ratio = dist_to_closest_seq[seq] / closest_seq_tree_dist
            ratios.append(seq_id_ratio / closest_seq_ratio)
    return ratios


def get_bootstrap_and_bts_scores(
    reduced_tree_files,
    full_tree_file,
):
    """
    Returns DataFrame branch_scores_df containing bts values for all edges in tree in full_tree_file
    """

    full_tree = Tree(full_tree_file)
    num_leaves = len(full_tree)

    # extract bootstrap support from full tree
    bootstrap_dict = {
        ",".join(sorted(node.get_leaf_names())): node.support
        for node in full_tree.iter_descendants()
        if not node.is_leaf()
    }
    bootstrap_df = pd.DataFrame(bootstrap_dict, index=["bootstrap_support"]).transpose()

    # initialise branch score dict with sets of leaves representing edges of
    # full_tree
    branch_scores = {}
    all_taxa = full_tree.get_leaf_names()
    branch_scores = {
        ",".join(sorted(node.get_leaf_names())): 0
        for node in full_tree.iter_descendants()
        if not node.is_leaf()
    }

    for treefile in reduced_tree_files:
        tree = Tree(treefile)
        seq_id = treefile.split("/")[-2]

        # collect bootstrap support and bts for all nodes in tree
        for node in tree.traverse("postorder"):
            if not node.is_leaf() and not node.is_root():
                # add bootstrap support values for seq_id
                # one edge in the full tree could correspond to two edges in the
                # reduced tree (leaf_str and leaf_str_extended below), if the pruned
                # taxon is attached on that edge. We hence need to add one for each of
                # those, if they are in branch_scores.
                edge_found = False
                leaf_list = node.get_leaf_names()
                leaf_str = ",".join(sorted(leaf_list))
                if leaf_str in branch_scores:
                    branch_scores[leaf_str] += 1
                    continue
                leaf_str_extended = ",".join(sorted(leaf_list + [seq_id]))
                if not edge_found and leaf_str_extended in branch_scores:
                    branch_scores[leaf_str_extended] += 1
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
                    continue
                leaf_str_extended = ",".join(complement_leaves + [seq_id])
                if leaf_str_extended in branch_scores:
                    branch_scores[leaf_str_extended] += 1
    # normalise BTS
    # Divide by num_leaves - 2 if edge is incident to cherry, as it cannot be present in
    # either of the two trees where one of those cherry leaves is pruned
    for branch_score in branch_scores:
        if branch_score.count(",") == 1 or branch_score.count(",") == num_leaves - 3:
            branch_scores[branch_score] *= 100 / (num_leaves - 2)  # normalise bts
        else:
            branch_scores[branch_score] *= 100 / num_leaves
        branch_scores[branch_score] = int(branch_scores[branch_score])
    branch_scores_df = pd.DataFrame(branch_scores, index=["bts"]).transpose()

    # sort both dataframes so we plot corresponding values correctly
    branch_scores_df = branch_scores_df.sort_values(
        by=list(branch_scores_df.columns)
    ).reset_index(drop=True)
    bootstrap_df = bootstrap_df.sort_values(by=list(bootstrap_df.columns)).reset_index(
        drop=True
    )
    merged_df = pd.concat([branch_scores_df, bootstrap_df], axis=1)

    return merged_df


for subdir in subdirs:
    print("Extracting reattachment statistics for ", subdir)

    # get all the data files we need
    subdir_full_tree_file = [f for f in full_tree_files if subdir in f][0]
    subdir_epa_result_files = [f for f in epa_result_files if subdir in f]
    subdir_restricted_tree_files = [f for f in restricted_tree_files if subdir in f]
    subdir_restricted_mldist_files = [f for f in restricted_mldist_files if subdir in f]
    subdir_full_mldist_file = [f for f in full_mldist_files if subdir in f][0]
    subdir_plot_csv = [f for f in plot_csvs if subdir in f][0]
    subdir_random_forest_csv = [f for f in random_forest_csvs if subdir in f][0]
    subdir_bootstrap_csv = [f for f in bootstrap_csvs if subdir in f][0]

    full_tree = Tree(subdir_full_tree_file)
    seq_ids = full_tree.get_leaf_names()

    output = []
    taxon_tii_list = []
    nj_tiis = get_nj_tiis(subdir_full_mldist_file, subdir_restricted_mldist_files)

    for seq_id in seq_ids:
        epa_file = [f for f in subdir_epa_result_files if "/" + seq_id + "/" in f][0]
        with open(epa_file, "r") as f:
            dict = json.load(f)
        # Compute RF TII
        restricted_tree_file = [
            f for f in subdir_restricted_tree_files if "/" + seq_id + "/" in f
        ][0]
        restricted_tree = Tree(restricted_tree_file)
        normalising_constant = 2 * len(full_tree) - 3
        rf_distance = full_tree.robinson_foulds(restricted_tree, unrooted_trees=True)[0]
        normalised_rf_distance = rf_distance / normalising_constant

        # get bootstrap support values in restricted tree
        bootstrap_list = [
            node.support
            for node in restricted_tree.traverse()
            if not node.is_leaf() and not node.is_root()
        ]

        placements = dict["placements"][0][
            "p"
        ]  # this is a list of lists, each containing information for one reattachment
        num_likely_reattachments = len(placements)
        # get values for which we need to iterate through all best reattachments
        reattached_trees = []
        for placement in placements:
            reattached_tree = get_reattached_tree(
                dict["tree"], placement[0], seq_id, placement[3], placement[4]
            )[0]
            reattached_trees.append(reattached_tree)
        reattachment_distances = get_reattachment_distances(
            restricted_tree, reattached_trees, seq_id
        )
        best_reattached_tree = get_reattached_tree(
            dict["tree"], placements[0][0], seq_id, placements[0][3], placements[0][4]
        )[0]
        # normalising constant for branch lenghts
        sum_branch_lengths = sum(
            [node.dist for node in best_reattached_tree.iter_descendants()]
        )
        # to avoid having an empty list, we set distance between reattachments to be 0.
        # Note that this is correct if three is only one best reattachment.
        if len(reattachment_distances) == 0:
            reattachment_distances = [0]

        best_placement = placements[0]
        edge_num = best_placement[0]
        likelihood = best_placement[1]
        like_weight_ratio = best_placement[2]
        distal_length = best_placement[3] / sum_branch_lengths
        pendant_length = best_placement[4] / sum_branch_lengths
        reattached_tree, reattachment_branch_length = get_reattached_tree(
            dict["tree"], best_placement[0], seq_id, distal_length, pendant_length
        )
        taxon_height = (
            calculate_taxon_height(reattached_tree, seq_id) / sum_branch_lengths
        )
        order_diff = get_order_of_distances_to_seq_id(
            seq_id, subdir_full_mldist_file, reattached_tree
        )
        dist_reattachment_low_bootstrap_node = (
            reattachment_distance_to_low_support_node(
                seq_id, reattached_tree, bootstrap_threshold=0.1
            )
        )
        seq_and_tree_dist_ratio = seq_and_tree_dist_diff(
            seq_id,
            subdir_full_mldist_file,
            reattached_tree,
        )
        dist_diff_reattachment_sibling = get_distance_reattachment_sibling(
            seq_id, subdir_full_mldist_file, reattached_tree
        )
        seq_distance_ratios_closest_seq = seq_distance_distribution_closest_seq(
            seq_id,
            subdir_full_mldist_file,
            reattached_tree,
        )
        # save all those summary statistics
        output.append(
            [
                seq_id + " " + str(rf_distance),
                likelihood,
                like_weight_ratio,
                reattachment_branch_length,
                pendant_length,
                rf_distance,
                normalised_rf_distance,
                taxon_height,
                num_likely_reattachments,
                bootstrap_list,
                nj_tiis[seq_id],
                order_diff,
                reattachment_distances,
                dist_reattachment_low_bootstrap_node,
                seq_and_tree_dist_ratio,
                dist_diff_reattachment_sibling,
                seq_distance_ratios_closest_seq,
            ]
        )

    df = pd.DataFrame(
        output,
        columns=[
            "seq_id",
            "likelihood",
            "like_weight_ratio",
            "reattachment_branch_length",
            "pendant_branch_length",
            "tii",
            "normalised_tii",
            "taxon_height",
            "num_likely_reattachments",
            "bootstrap",
            "nj_tii",
            "order_diff",
            "reattachment_distances",
            "dist_reattachment_low_bootstrap_node",
            "seq_and_tree_dist_ratio",
            "dist_diff_reattachment_sibling",
            "seq_distance_ratios_closest_seq",
        ],
    )
    df.to_csv(subdir_plot_csv)

    # replace lists by mean and standard deviation for training random forest
    def calculate_mean_std(cell):
        if isinstance(cell, list) or isinstance(cell, np.ndarray):
            # cell is already a list or a numpy array
            arr = np.array(cell)
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            return mean_val, std_val
        else:
            # Return NaN if the cell is not a list or array
            return np.nan, np.nan

    # Loop through each column in the DataFrame
    for col in df.columns:
        if (
            df[col].dtype == object and col != "seq_id"
        ):  # Apply only to columns with object type
            # Create new columns for mean and standard deviation
            df[f"{col}_mean"], df[f"{col}_std"] = zip(*df[col].map(calculate_mean_std))
            df = df.drop(columns=[col])
    df.to_csv(subdir_random_forest_csv)

    # get bootstap and bts values
    merged_df = get_bootstrap_and_bts_scores(
        subdir_restricted_tree_files,
        subdir_full_tree_file,
    )
    merged_df.to_csv(subdir_bootstrap_csv)
