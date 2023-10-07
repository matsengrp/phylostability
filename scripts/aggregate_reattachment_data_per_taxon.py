import pandas as pd
import numpy as np
import sys
from ete3 import Tree

full_tree_file = snakemake.input.full_treefile
all_tree_files = snakemake.input.treefiles
reduced_tree_files = snakemake.input.reduced_treefile
reduced_tree_mlfiles = snakemake.input.reduced_tree_mlfile
seq_ids = snakemake.params.seq_ids
edge_ids = snakemake.params.edges
taxon_dfs = snakemake.input.taxon_dictionary
output_file = snakemake.output.output_csv
reattachment_distances_csv = snakemake.output.reattachment_distance_csv


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


def attachment_branch_length_proportion(node, seq_id, above=True, topology_only=False):
    """
    This function returns the proportion of the branch above
    the input ete3.TreeNode "node" at which the new taxon was attached.

    Note: this function should be called for any TreeNode in the augmented
    topology that can be interpreted as the child node of an attachment
    edge location in the reduced topology.

    Every node satisfying this description will have at least 2 ancestors:
      - the new parent node of this node and the reattached taxon
      - the "grandparent node" that corresponds to the parent node of the original
        reattachment edge location in the reduced topology.
    """
    parent = node.up
    if len(parent.get_ancestors()) > 0:
        gp = parent.up
    else:
        gp = [
            x
            for x in parent.get_children()
            if (x != node and seq_id not in [l.name for l in x.get_leaves()])
        ][0]
    leaf_below_node = node.get_leaves()[0]
    parent_dist = ete_dist(node, parent, topology_only)
    gp_dist = ete_dist(parent, gp, topology_only)
    if above:
        return gp_dist / (parent_dist + gp_dist)
    else:
        return parent_dist / (parent_dist + gp_dist)


def dist(n1, n2, alt_n1, alt_n2, seq_id, topology_only=False):
    """
    Returns the path distance between two nodes n1 and n2, adjusted by the
    amount of the path that would be removed by reattaching a taxon at an
    optimal location along the edge above each of n1 and n2 (these optimal
    locations, and their respective edge lengths, are computed on another
    tree, and so they correspond to the "alternative" nodes alt_n1 and alt_n2 resp.
    """
    # no correction for branch lengths needed if we want topological distance
    if topology_only:
        return ete_dist(n1, n2, topology_only)
    # get the lengths of the branches above n1 and n2
    n1_branch_len = ete_dist(n1, n1.up, topology_only)
    n2_branch_len = ete_dist(n2, n2.up, topology_only)
    # adjust the distance calculation based on how far along each branch the reattachment happens
    if n1 in n2.get_ancestors():
        return (
            ete_dist(n2, n1, topology_only)
            + n1_branch_len
            * attachment_branch_length_proportion(alt_n1, seq_id, False, topology_only)
            - n2_branch_len
            * attachment_branch_length_proportion(alt_n2, seq_id, False, topology_only)
        )
    elif n2 in n1.get_ancestors():
        return (
            ete_dist(n1, n2, topology_only)
            - n1_branch_len
            * attachment_branch_length_proportion(alt_n1, seq_id, False, topology_only)
            + n2_branch_len
            * attachment_branch_length_proportion(alt_n2, seq_id, False, topology_only)
        )
    else:
        return (
            ete_dist(n1, n2, topology_only)
            - n1_branch_len
            * attachment_branch_length_proportion(alt_n1, seq_id, False, topology_only)
            - n2_branch_len
            * attachment_branch_length_proportion(alt_n2, seq_id, False, topology_only)
        )


def get_likelihood(input_file):
    likelihood = 0
    ll_str = "Log-likelihood"
    with open(input_file, "r") as f:
        for line in f.readlines():
            if ll_str in line:
                likelihood = float(line.split(": ")[-1].split(" ")[0].strip())
                break
    return likelihood


def tree_branch_length_sum(tree):
    result = 0
    for node in tree.traverse("postorder"):
        if not node.is_root():
            result += ete_dist(node, node.up)
    return result


with open(full_tree_file, "r") as f:
    full_tree = Tree(f.readlines()[0])

all_taxa_df = {}
for idx, seq_id in enumerate(seq_ids):
    tree_files = [x for x in all_tree_files if seq_id in x]
    reduced_tree_file = reduced_tree_files[idx]
    reduced_tree_mlfile = reduced_tree_mlfiles[idx]
    df = pd.read_csv(taxon_dfs[idx], index_col=0)

    # load tree on reduced taxon set
    with open(reduced_tree_file, "r") as f:
        main_tree = Tree(f.readlines()[0])

    # create a lookup dictionary between the main tree's nodes and its copy as the child node
    # of a reattachment edge for the corresponding tree in one of the tree_files
    node_lookup_dictionary = {
        n: {"leafset": set(), "node": n, "likelihood_ratio": 0, "likelihood": 0}
        for n in main_tree.traverse("postorder")
    }

    # leafset is used to create correspondence between nodes in different trees
    node_lookup_dictionary.pop(main_tree.get_tree_root())
    for node, node_lookup in node_lookup_dictionary.items():
        node_lookup["leafset"] = set([l.name for l in node.get_leaves()])

    # load all of the reattached-taxon trees for the specific seq_id
    for tree_file in tree_files:
        with open(tree_file, "r") as f:
            other_tree = Tree(f.readlines()[0])

        # get the "edge" wildcard corrdesponding to the current tree_file
        reattachment_edge = tree_file.split("edge_")[-1].split(".")[0]
        seq_id_taxon = other_tree & seq_id
        reattachment_node = seq_id_taxon.get_sisters()[0]
        reattachment_node_lfst = set([l.name for l in reattachment_node.get_leaves()])

        # find the node corresponding to 'reattachment_node' in 'main_tree', and update dictionary entry
        for node, node_lookup in node_lookup_dictionary.items():
            lfst = node_lookup["leafset"]
            if lfst == reattachment_node_lfst:
                node_lookup["likelihood_ratio"] = df.loc[
                    seq_id + "_" + reattachment_edge, "likelihood_ratio"
                ]
                node_lookup["likelihood"] = df.loc[
                    seq_id + "_" + reattachment_edge, "likelihood"
                ]
                node_lookup["node"] = reattachment_node
                break
    # Calculate EDPL for the taxon seq_id:
    edpl = 0
    reattachment_distances = {}  # save distance matrix of reattachments
    reattachment_topological_distances = {}
    reattachment_dist_file = [
        csv for csv in reattachment_distances_csv if seq_id in csv
    ][0]
    edge_likelihoods = []
    for n1, node_lookup1 in node_lookup_dictionary.items():
        node_str = ",".join(sorted(list([l.name for l in n1.get_leaves()])))
        reattachment_distances[node_str] = []
        reattachment_topological_distances[node_str] = []
        edge_likelihoods.append(node_lookup1["likelihood"])
        for n2, node_lookup2 in node_lookup_dictionary.items():
            if n1 != n2:
                dist_n1_n2 = dist(
                    n1,
                    n2,
                    node_lookup1["node"],
                    node_lookup2["node"],
                    seq_id,
                    topology_only=False,
                )
                reattachment_distances[node_str].append(dist_n1_n2)
                reattachment_topological_distances[node_str].append(
                    dist(
                        n1,
                        n2,
                        node_lookup1["node"],
                        node_lookup2["node"],
                        seq_id,
                        topology_only=True,
                    )
                )
                edpl += (
                    dist_n1_n2
                    * node_lookup1["likelihood_ratio"]
                    * node_lookup2["likelihood_ratio"]
                )
            else:
                reattachment_distances[node_str].append(np.nan)
                reattachment_topological_distances[node_str].append(np.nan)

    reattachment_dist_df = pd.DataFrame(reattachment_distances)
    reattachment_topological_dist_df = pd.DataFrame(reattachment_topological_distances)

    # Check that all diagonal entries in reattachment_dist_df are 0 to make sure we get
    # correct distance matrix
    diagonal_values = np.diag(reattachment_dist_df.values)
    if not np.all(np.isnan(diagonal_values)):
        print("Error: reattachment distance matrix has non-zero diagonal entries.")
        sys.exit(1)
    reattachment_dist_df["likelihoods"] = edge_likelihoods
    reattachment_topological_dist_df["likelihoods"] = edge_likelihoods

    # save everything in corresponding DataFrames
    reattachment_dist_df.to_csv(reattachment_dist_file)
    reattachment_topological_dist_df.to_csv(
        reattachment_dist_file[:-4] + "_topological.csv"
    )
    edpl /= tree_branch_length_sum(main_tree)
    tii = main_tree.robinson_foulds(full_tree, unrooted_trees=True)[0]
    all_taxa_df[seq_id] = [edpl, get_likelihood(reduced_tree_mlfile), tii]

all_taxa_df = pd.DataFrame(all_taxa_df).transpose()
all_taxa_df.columns = ["edpl", "likelihood", "tii"]
all_taxa_df.to_csv(output_file)
