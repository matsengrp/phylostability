from ete3 import Tree
import pandas as pd
import json
import re
import numpy as np
import itertools

from utils import *

epa_result_files = snakemake.input.epa_results
full_tree_file = snakemake.input.full_tree
restricted_trees = snakemake.input.restricted_trees

full_mldist_file = snakemake.input.full_mldist_file
restricted_mldist_files = snakemake.input.restricted_mldist_files

reattached_tree_files = snakemake.output.reattached_trees
output_csv = snakemake.output.output_csv

seq_ids = snakemake.params.seq_ids


def get_reattached_tree(newick_str, edge_num, seq_id, distal_length, pendant_length):
    """
    From output information of epa-ng, compute reattached tree.
    Returns reattached tree and branch length of branch on which we reattach
    """
    # replace labels of internal nodes by names and keep support values for each internal node in node_support_dict
    newick_str_split = newick_str.split(")")
    new_newick_str = ""
    current_number = 1
    int_node_dict = {}
    for s in newick_str_split:
        # Check if the string starts with an integer followed by ":"
        match = re.match(r"(^\d+):", s)
        if match:
            int_node_dict[str(current_number)] = match.group(1)
            # Replace the integer with the current_number and increment it
            s = re.sub(r"^\d+", str(current_number), s)
            current_number += 1
        s += (
            ")"  # add bracket back in that we deleted when splitting the string earlier
        )
        new_newick_str += s
    new_newick_str = new_newick_str[:-1]  # delete extra ) at end of string

    # find label of node above which we reattach edge
    pattern = re.compile(r"([\w.]+):[0-9.]+\{" + str(edge_num) + "\}")
    match = pattern.search(new_newick_str)
    sibling_node_id = match.group(1)
    if sibling_node_id.isdigit():
        sibling_node_id = str(int(sibling_node_id))

    # delete edge numbers in curly brackets
    ete_newick_str = re.sub(r"\{\d+\}", "", new_newick_str)

    # add new node to tree
    tree = Tree(ete_newick_str, format=2)
    for node in [node for node in tree.iter_descendants() if not node.is_leaf()]:
        # label at internal nodes are interpreted as support, we need to set names to be that value
        node.name = str(int(node.support))
    sibling = tree.search_nodes(name=sibling_node_id)[0]
    reattachment_branch_length = sibling.dist

    dist_from_parent = reattachment_branch_length - distal_length
    # support of new internal node shall be 0
    new_internal_node = sibling.up.add_child(name=0, dist=dist_from_parent)
    sibling.detach()
    new_internal_node.add_child(sibling, dist=distal_length)
    new_internal_node.add_child(name=seq_id, dist=pendant_length)

    # add support values back to tree and save reattached tree (new internal node gets support 1.0)
    for node in [node for node in tree.iter_descendants() if not node.is_leaf()]:
        if str(int(node.name)) in int_node_dict:
            n = str(int(node.name))
            node.name = int_node_dict[n]
            node.support = int_node_dict[n]
    return tree, reattachment_branch_length


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
    for file in restricted_mldist_files:
        seq_id = file.split("/")[-2]
        restricted_mldist = get_ml_dist(file)
        restricted_nj_tree = compute_nj_tree(restricted_mldist)
        tiis[seq_id] = full_nj_tree.robinson_foulds(
            restricted_nj_tree, unrooted_trees=True
        )[0]
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
        reattachment_distances.append(ete_dist(node1, node2, topology_only=True))
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
        if node.support < q and dist < min_dist_found:
            min_dist_found = dist
    return min_dist_found


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
    distances = []
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


full_tree = Tree(full_tree_file)

output = []
taxon_tii_list = []

for seq_id in seq_ids:
    epa_file = [f for f in epa_result_files if "/" + seq_id + "/" in f][0]
    with open(epa_file, "r") as f:
        dict = json.load(f)

    # compute and safe reattached tree
    tree_file = [f for f in reattached_tree_files if "/" + seq_id + "/" in f][0]

    # Compute RF TII
    restricted_tree_file = [f for f in restricted_trees if "/" + seq_id + "/" in f][0]
    restricted_tree = Tree(restricted_tree_file)
    rf_distance = full_tree.robinson_foulds(restricted_tree, unrooted_trees=True)[0]

    # get bootstrap support values in restricted tree
    bootstrap_list = [
        node.support
        for node in restricted_tree.traverse()
        if not node.is_leaf() and not node.is_root()
    ]

    placements = dict["placements"][0][
        "p"
    ]  # this is a list of lists, each containing information for one reattachment
    newick_trees = []
    num_likely_reattachments = len(placements)
    nj_tiis = get_nj_tiis(full_mldist_file, restricted_mldist_files)

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
    # to avoid having an empty list, we set distance between reattachments to be 0.
    # Note that this is correct if three is only one best reattachment.
    if len(reattachment_distances) == 0:
        reattachment_distances = [0]

    best_placement = placements[0]
    edge_num = best_placement[0]
    likelihood = best_placement[1]
    like_weight_ratio = best_placement[2]
    distal_length = best_placement[3]
    pendant_length = best_placement[4]
    reattached_tree, reattachment_branch_length = get_reattached_tree(
        dict["tree"], best_placement[0], seq_id, distal_length, pendant_length
    )
    taxon_height = calculate_taxon_height(reattached_tree, seq_id)
    order_diff = get_order_of_distances_to_seq_id(
        seq_id, full_mldist_file, reattached_tree
    )
    dist_reattachment_low_bootstrap_node = reattachment_distance_to_low_support_node(
        seq_id, reattached_tree, bootstrap_threshold=0.1
    )
    seq_and_tree_dist_ratio = seq_and_tree_dist_diff(
        seq_id,
        full_mldist_file,
        reattached_tree,
    )
    dist_diff_reattachment_sibling = get_distance_reattachment_sibling(
        seq_id, full_mldist_file, reattached_tree
    )
    seq_distance_ratios_closest_seq = seq_distance_distribution_closest_seq(
        seq_id,
        full_mldist_file,
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
    newick_trees.append(reattached_tree.write(format=0))

    with open(tree_file, "w") as f:
        for newick_str in newick_trees:
            f.write(newick_str + "\n")
df = pd.DataFrame(
    output,
    columns=[
        "seq_id",
        "likelihood",
        "like_weight_ratio",
        "reattachment_branch_length",
        "pendant_branch_length",
        "tii",
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
df.to_csv(output_csv)
