import pandas as pd
from ete3 import Tree

full_tree_file = snakemake.input.full_treefile
all_tree_files = snakemake.input.treefiles
reduced_tree_files = snakemake.input.reduced_treefile
reduced_tree_mlfiles = snakemake.input.reduced_tree_mlfile
seq_ids = snakemake.params.seq_ids
edge_ids = snakemake.params.edges
taxon_dfs = snakemake.input.taxon_dictionary
output_file = snakemake.output.output_csv

def ete_dist(node1, node2, topology_only=False):
    if node2 in node1.get_ancestors():
        leaf = node1.get_leaves()[0]
        return node2.get_distance(leaf, topology_only=topology_only) - node1.get_distance(leaf, topology_only=topology_only)
    else:
        leaf = node2.get_leaves()[0]
        return node1.get_distance(leaf, topology_only=topology_only) - node2.get_distance(leaf, topology_only=topology_only)

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
    parent = node.get_ancestors()[0]
    if len(parent.get_ancestors()) > 0:
        gp = parent.get_ancestors()[0]
    else:
        gp = [x for x in parent.get_children() if (x != node and seq_id not in [l.name for l in x.get_leaves()])][0]
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
    # get the lengths of the branches above n1 and n2
    n1_branch_len = ete_dist(n1, n1.get_ancestors()[0], topology_only)
    n2_branch_len = ete_dist(n2, n2.get_ancestors()[0], topology_only)
    # adjust the distance calculation based on how far along each branch the reattachment happens

    if n1 in n2.get_ancestors():
        return ete_dist(n2, n1, topology_only) \
             + n1_branch_len*attachment_branch_length_proportion(alt_n1, seq_id, True, topology_only) \
             - n2_branch_len*attachment_branch_length_proportion(alt_n2, seq_id, False, topology_only)
    elif n2 in n1.get_ancestors():
        return ete_dist(n1, n2, topology_only) \
             - n1_branch_len*attachment_branch_length_proportion(alt_n1, seq_id, False, topology_only) \
             + n2_branch_len*attachment_branch_length_proportion(alt_n2, seq_id, True, topology_only)
    else:
        return ete_dist(n1, n2, topology_only) \
             - n1_branch_len*attachment_branch_length_proportion(alt_n1, seq_id, False, topology_only) \
             - n2_branch_len*attachment_branch_length_proportion(alt_n2, seq_id, False, topology_only)

def get_likelihood(input_file):
    likelihood = 0
    ll_str = "Log-likelihood"
    with open(input_file, "r") as f:
        for line in f.readlines():
            if ll_str in line:
                likelihood = float(line.split(": ")[-1].split(" ")[0].strip())
                break
    return likelihood


with open(full_tree_file, "r") as f:
    full_tree = Tree(f.readlines()[0])

all_taxa_df = {}
for idx, seq_id in enumerate(seq_ids):
    tree_files = [x for x in all_tree_files if seq_id in x]
    reduced_tree_file = reduced_tree_files[idx]
    reduced_tree_mlfile = reduced_tree_mlfiles[idx]
    df = pd.read_csv(taxon_dfs[idx], index_col=0)

    # load tree on full taxon set
    with open(reduced_tree_file, "r") as f:
        main_tree = Tree(f.readlines()[0])

    # create a lookup dictionary between the main tree's nodes and its copy as the child node 
    # of a reattachment edge for the corresponding tree in one of the tree_files
    node_lookup_dictionary = {n:{"leafset":set(), "node":n, "likelihood":0} for n in main_tree.traverse("postorder")}

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
        reattachment_node = [x for x in seq_id_taxon.get_ancestors()[0].get_children() if x != seq_id_taxon][0]
        reattachment_node_lfst = set([l.name for l in reattachment_node.get_leaves()])

        # find the node corresponding to 'reattachment_node' in 'main_tree', and update dictionary entry
        for node, node_lookup in node_lookup_dictionary.items():
            lfst = node_lookup["leafset"]
            if lfst == reattachment_node_lfst:
                node_lookup["likelihood"] = df.loc[seq_id + "_" + reattachment_edge, "likelihood_ratio"]
                node_lookup["node"] = reattachment_node
                break

    # Calculate EDPL for the taxon seq_id:
    edpl = 0
    for n1, node_lookup1 in node_lookup_dictionary.items():
        for n2, node_lookup2 in node_lookup_dictionary.items():
            if n1 != n2:
                edpl += dist(n1, n2, node_lookup1["node"], node_lookup2["node"], seq_id) \
                      * node_lookup1["likelihood"] \
                      * node_lookup2["likelihood"]

    edpl /= get_likelihood(reduced_tree_mlfile)
    tii = main_tree.robinson_foulds(full_tree, unrooted_trees = True)[0]
    all_taxa_df[seq_id] = [edpl, get_likelihood(reduced_tree_mlfile), tii]

all_taxa_df = pd.DataFrame(all_taxa_df).transpose()
all_taxa_df.columns = ["edpl", "likelihood", "tii"]
all_taxa_df.to_csv(output_file)
