from ete3 import Tree
import pandas as pd
import json
import numpy as np
import itertools

from utils import *

dynamic_input = snakemake.input.dynamic_input
full_tree_file = snakemake.input.full_tree
full_mldist_file = snakemake.input.full_mldist_file

plot_csv = snakemake.output.plot_csv
random_forest_csv = snakemake.output.random_forest_csv
bootstrap_csv = snakemake.output.bootstrap_csv


def calculate_taxon_height(input_tree, taxon_name):
    """
    Return distance of reattachment position to nearest leaf (that is not the reattached leaf), normalised by maximum distance between any two leaves divided by two
    """
    taxon = input_tree & taxon_name
    taxon_parent = taxon.up
    taxon_height = min(
        [
            ete_dist(taxon_parent, leaf, topology_only=False)
            for leaf in input_tree.get_leaves()
            if leaf != taxon
        ]
    )
    normalising_constant = max(
        [
            ete_dist(leaf1, leaf2, topology_only=False)
            for leaf1, leaf2 in itertools.combinations(input_tree.get_leaves(), 2)
        ]
    )
    return taxon_height / normalising_constant


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
    Get difference in full_tree and sequence distance for sequences of taxon1 and taxon2
    where distance of taxon1 to seq_id is smaller than taxon2 to seq_id in best reattached
    full_tree, but greater in terms of sequence distance, for each possible seq_id.
    If there are not two taxa with this property, we return [0]
    """
    mldist = get_ml_dist(mldist_file)
    max_mldist = mldist.max().max()
    dist_diff_list = []

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
    q = np.quantile(all_bootstraps, 1)
    for leaf1, leaf2 in itertools.combinations(leaves, 2):
        mrca = reattached_tree.get_common_ancestor([leaf1, leaf2])
        if mrca.support > q or mrca.support == 1.0:
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
            dist_diff_list.append(difference)
        elif (
            tree_dist_leaf2 / tree_dist_leaf1 < 1
            and seq_dist_leaf2 / seq_dist_leaf1 > 1
        ):
            difference = (
                seq_dist_leaf2 / seq_dist_leaf1 - tree_dist_leaf2 / tree_dist_leaf1
            )
            dist_diff_list.append(difference)
    return dist_diff_list


def get_reattachment_distances(reduced_tree, reattachment_trees, seq_id):
    """
    Return list of distances between best reattachment locations given in reattachment_trees
    """
    if len(reattachment_trees) == 1:
        return [0]
    reattachment_node_list = []

    # maximum reattachment distance could reach is max dist between any two leaves in the
    # smaller full_tree
    max_reattachment_dist = max(
        [
            leaf1.get_distance(leaf2, topology_only=True)
            for leaf1, leaf2 in itertools.combinations(reduced_tree.get_leaves(), 2)
        ]
    )
    for full_tree in reattachment_trees:
        # cluster is set of leaves below lower node of reattachment edge
        cluster = full_tree.search_nodes(name=seq_id)[0].up.get_leaf_names()
        cluster.remove(seq_id)
        if len(cluster) == 1:  # reattachment above leaf
            reattachment_node = full_tree.search_nodes(name=cluster[0])[0]
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


def normalised_dist_closest_low_bootstrap_node(node, tree, threshold=70):
    """
    Compute topological distance to closet node with bootstrap support < 70
    If edge=True, we take minimum of distance of node and node.up to low bootstrap
    support node.
    """
    low_bootstrap_nodes = [
        n
        for n in tree.traverse()
        if not n.is_leaf() and not n.is_root() and n != node and n.support < threshold
    ]
    if len(low_bootstrap_nodes) == 0:
        return np.nan
    min_dist = min([ete_dist(n, node, topology_only=True) for n in low_bootstrap_nodes])
    max_dist = max(
        [
            ete_dist(n, node, topology_only=True)
            for n in tree.traverse()
            if not n.is_leaf()
        ]
    )
    return min_dist / max_dist


def reattachment_distance_to_low_support_node(seq_id, reattached_tree):
    """
    Compute (topological) distance of reattachment position in best_reattached_tree to
    nearest low bootstrap node for each seq_id.
    """
    reattachment_node = reattached_tree.search_nodes(name=seq_id)[0].up
    dist = normalised_dist_closest_low_bootstrap_node(
        reattachment_node, reattached_tree
    )
    return dist


def changed_edge_dist_to_low_bootstrap(reduced_tree, full_tree):
    """
    Compute mean distance of edges that are in reduced tree but not full tree
    to low bootstrap support node (in reduced tree).
    We set distance of an edge to be the distance of the closest node of the
    considered edge to a low bootstrap node.
    Returns 0 if reduced_tree and full_tree have same topology
    """
    rf_output = full_tree.robinson_foulds(reduced_tree, unrooted_trees=True)
    changed_edges = rf_output[3] - rf_output[4]
    if len(changed_edges) == 0:
        return np.nan
    dist_list = []
    for set in changed_edges:
        # find lower node of changed edge (node)
        cluster1 = set[0]
        node = reduced_tree.get_common_ancestor(cluster1)
        if node.is_root():
            cluster2 = set[1]
            node = reduced_tree.get_common_ancestor(cluster2)
        # find closest low bootstrap node
        dist_to_low_bootstrap = normalised_dist_closest_low_bootstrap_node(
            node, reduced_tree
        )
        dist_list.append(dist_to_low_bootstrap)
    return sum(dist_list) / len(dist_list)


def seq_and_tree_dist_diff(
    seq_id,
    mldist_file,
    reattached_tree,
):
    """
    Get difference and ratio of distance between seq_id and other sequences in alignment
    to corresponding distance in best reattached full_tree.
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
    full_tree to average distance of reattached sequences and sequences in S'.
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
    return avg_new_node_distance / avg_sibling_distance


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
    Returns DataFrame branch_scores_df containing bts values for all edges in full_tree in full_tree_file
    """

    full_tree = Tree(full_tree_file)
    num_leaves = len(full_tree)

    # extract bootstrap support from full full_tree
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
    # get bootstrap dict that contains min bootstrap values for edge we consider for bts
    bootstrap_per_bts_dict = {
        ",".join(sorted(node.get_leaf_names())): min([node.support, node.up.support])
        for node in full_tree.iter_descendants()
        if not node.is_leaf() and not node.up.is_root()
    }
    for child in [child for child in full_tree.get_children() if not child.is_leaf()]:
        bootstrap_per_bts_dict[",".join(sorted(child.get_leaf_names()))] = child.support

    for treefile in reduced_tree_files:
        full_tree = Tree(treefile)
        seq_id = treefile.split("/")[-2]

        # collect bootstrap support and bts for all nodes in full_tree
        for node in full_tree.traverse("postorder"):
            if not node.is_leaf() and not node.is_root():
                # add bootstrap support values for seq_id
                # one edge in the full full_tree could correspond to two edges in the
                # reduced full_tree (leaf_str and leaf_str_extended below), if the pruned
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
                # this could happen if rooting of full_tree is different to that of full_tree
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
    branch_scores_df = pd.DataFrame(
        list(branch_scores.items()), columns=["edges", "bts"]
    )
    bootstrap_per_bts_df = pd.DataFrame(
        list(bootstrap_per_bts_dict.items()), columns=["edges", "bootstrap"]
    )

    merged_df = pd.merge(branch_scores_df, bootstrap_per_bts_df, on="edges")

    return merged_df


def get_rf_radius(full_tree, reduced_tree, seq_id):
    """ "
    Return maximum distance of split that's present in full_tree but not
    reduced_tree to reattachment position.
    Normalise by dividing by maximum possible distance of reattachment
    to any edge.
    """
    rf_output = full_tree.robinson_foulds(reduced_tree, unrooted_trees=True)
    changed_edges = rf_output[3] - rf_output[4]
    rf_radius = 0
    seq_id_leaf = full_tree.search_nodes(name=seq_id)[0]
    reattachment_position = seq_id_leaf.up

    for set in changed_edges:
        # find lower node of changed edge (node)
        cluster1 = set[0]
        node = full_tree.get_common_ancestor(cluster1)
        if node.is_root():
            cluster2 = set[1]
            node = full_tree.get_common_ancestor(cluster2)
        node_dist = ete_dist(node, reattachment_position, topology_only=True)
        node_up_dist = ete_dist(node.up, reattachment_position, topology_only=True)
        dist = max(node_dist, node_up_dist)
        if dist > rf_radius:
            rf_radius = dist
    normalising_constant = max(
        [
            ete_dist(node, reattachment_position, topology_only=True)
            for node in full_tree.traverse()
            if not node.is_leaf()
        ]
    )
    return rf_radius / normalising_constant


# get all the data files we need
full_tree = Tree(full_tree_file)
seq_ids = full_tree.get_leaf_names()
n = len(seq_ids)
epa_result_files = dynamic_input[0:n]
restricted_tree_files = dynamic_input[n : 2 * n]
restricted_mldist_files = dynamic_input[2 * n : 3 * n]

output = []
taxon_tii_list = []
nj_tiis = get_nj_tiis(full_mldist_file, restricted_mldist_files)
tii_normalising_constant = 2 * (len(full_tree) - 3)
nj_tiis = {s: nj_tiis[s] / tii_normalising_constant for s in nj_tiis}

for seq_id in seq_ids:
    epa_file = [f for f in epa_result_files if "/" + seq_id + "/" in f][0]
    with open(epa_file, "r") as f:
        dict = json.load(f)
    # Compute RF TII
    restricted_tree_file = [
        f for f in restricted_tree_files if "/" + seq_id + "/" in f
    ][0]
    restricted_tree = Tree(restricted_tree_file)
    rf_distance = full_tree.robinson_foulds(restricted_tree, unrooted_trees=True)[0]
    normalised_rf_distance = rf_distance / tii_normalising_constant

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
    branch_length_normalisation = sum_branch_lengths / n
    # to avoid having an empty list, we set distance between reattachments to be 0.
    # Note that this is correct if three is only one best reattachment.
    if len(reattachment_distances) == 0:
        reattachment_distances = [0]

    best_placement = placements[0]
    edge_num = best_placement[0]
    likelihood = best_placement[1]
    like_weight_ratio = best_placement[2]
    distal_length = best_placement[3]
    pendant_length = best_placement[4] / branch_length_normalisation
    reattached_tree, reattachment_branch_length = get_reattached_tree(
        dict["tree"],
        best_placement[0],
        seq_id,
        best_placement[3],
        best_placement[4],
    )
    distal_length = (
        min(distal_length, reattachment_branch_length - distal_length)
        / branch_length_normalisation
    )
    reattachment_branch_length = (
        reattachment_branch_length / branch_length_normalisation
    )
    taxon_height = calculate_taxon_height(reattached_tree, seq_id)
    if taxon_height < 0:
        print("ERROR calculating taxon height for ", full_tree_file, " taxon: ", seq_id)
    norm_taxon_height = taxon_height
    # order_diff = get_order_of_distances_to_seq_id(
    #     seq_id, full_mldist_file, reattached_tree
    # )
    dist_reattachment_low_bootstrap_node = reattachment_distance_to_low_support_node(
        seq_id, reattached_tree
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
    rf_radius = get_rf_radius(full_tree, restricted_tree, seq_id)
    change_to_low_bootstrap_dist = changed_edge_dist_to_low_bootstrap(
        restricted_tree, full_tree
    )
    # save all those summary statistics
    output.append(
        [
            seq_id + " " + str(rf_distance),
            n,
            likelihood,
            like_weight_ratio,
            reattachment_branch_length,
            distal_length,
            pendant_length,
            rf_distance,
            normalised_rf_distance,
            taxon_height,
            norm_taxon_height,
            num_likely_reattachments,
            bootstrap_list,
            nj_tiis[seq_id],
            # order_diff,
            reattachment_distances,
            dist_reattachment_low_bootstrap_node,
            seq_and_tree_dist_ratio,
            dist_diff_reattachment_sibling,
            seq_distance_ratios_closest_seq,
            rf_radius,
            change_to_low_bootstrap_dist,
        ]
    )

df = pd.DataFrame(
    output,
    columns=[
        "seq_id",
        "num_leaves",
        "likelihood",
        "like_weight_ratio",
        "reattachment_branch_length",
        "distal_length",
        "pendant_branch_length",
        "tii",
        "normalised_tii",
        "taxon_height",
        "norm_taxon_height",
        "num_likely_reattachments",
        "bootstrap",
        "nj_tii",
        # "order_diff",
        "reattachment_distances",
        "dist_reattachment_low_bootstrap_node",
        "seq_and_tree_dist_ratio",
        "dist_diff_reattachment_sibling",
        "seq_distance_ratios_closest_seq",
        "rf_radius",
        "change_to_low_bootstrap_dist",
    ],
)
df.to_csv(plot_csv)

# update df to save for random forest
# drop non-normalised taxon height -- we only wanted it for plotting
df = df.drop(columns=["taxon_height"])
# rename normalised taxon height column
df = df.rename(columns={"norm_taxon_height": "taxon_height"})


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
df.to_csv(random_forest_csv)

# get bootstap and bts values
merged_df = get_bootstrap_and_bts_scores(
    restricted_tree_files,
    full_tree_file,
)
merged_df.to_csv(bootstrap_csv)
