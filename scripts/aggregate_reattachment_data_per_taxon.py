import pandas as pd
from ete3 import Tree

all_tree_files = snakemake.input.treefiles
reduced_tree_files = snakemake.input.reduced_treefile
seq_ids = snakemake.params.seq_ids
edge_ids = snakemake.params.edges
taxon_dfs = snakemake.input.taxon_dictionary
output_file = snakemake.output.output_csv

def attachment_branch_length_proportion(node, above=True):
    """
       This function eturns the proportion of the branch above
       the input ete3.TreeNode "node" at which the new taxon was attached.

       Note: this function should be called for any TreeNode in the augmented
       topology that can be interpreted as the child node of an attachment 
       edge location in the reduced topology.

       Every node satisfying this description will have at least 2 ancestors: 
         - the new parent node of this node and the reattached taxon
         - the "grandparent node" that corresponds to the parent node of the original 
           reattachment edge location in the reduced topology.
    """
    if len(node.get_ancestors()) < 2:
        return 0
    parent_dist = node.get_ancestors()[0].get_distance(node)
    gp_dist = node.get_ancestors()[1].get_distance(node.get_ancestors()[0])
    if above:
        return gp_dist / (parent_dist + gp_dist)
    else:
        return parent_dist / (parent_dist + gp_dist)

def dist(n1, n2, alt_n1, alt_n2):
    """
        Returns the path distance between two nodes n1 and n2, adjusted by the 
        amount of the path that would be removed by reattaching a taxon at an
        optimal location along the edge above each of n1 and n2 (these optimal
        locations, and their respective edge lengths, are computed on another 
        tree, and so they correspond to the "alternative" nodes alt_n1 and alt_n2 resp.
    """
    # get the lengths of the branches above n1 and n2
    n1_branch_len = n1.get_ancestors()[0].get_distance(n1)
    n2_branch_len = n2.get_ancestors()[0].get_distance(n2)
    # adjust the distance calculation based on how far along each branch the reattachment happens
    if n1 in n2.get_ancestors():
        return n1.get_distance(n2) \
             + n1_branch_len*attachment_branch_length_proportion(alt_n1, True) \
             - n2_branch_len*attachment_branch_length_proportion(alt_n2, False)
    elif n2 in n1.get_ancestors():
        return n1.get_distance(n2) \
             - n1_branch_len*attachment_branch_length_proportion(alt_n1, False) \
             + n2_branch_len*attachment_branch_length_proportion(alt_n2, True)
    else:
        return n1.get_distance(n2) \
             - n1_branch_len*attachment_branch_length_proportion(alt_n1, False) \
             - n2_branch_len*attachment_branch_length_proportion(alt_n2, False)


all_taxa_df = {}
for idx, seq_id in enumerate(seq_ids):
    tree_files = [x for x in all_tree_files if seq_id in x]
    reduced_tree_file = reduced_tree_files[idx]
    df = pd.read_csv(taxon_dfs[idx])

    # load tree on full taxon set
    with open(reduced_tree_file, "r") as f:
        main_tree = Tree(f.readlines()[0])

    # create a lookup dictionary between the main tree's nodes and its copy as the child node 
    # of a reattachment edge for the corresponding tree in one of the tree_files
    node_lookup_dictionary = {n:{"leafset":set(), "node":n, "likelihood":0} for n in main_tree.traverse("postorder")}

    # leafset is used to create correspondence between nodes in different trees
    node_lookup_dictionary.pop(main_tree.get_tree_root())
    for node, node_lookup in node_lookup_dictionary.items():
        lfst = node_lookup["leafset"]
        lfst = set([l.name for l in node.get_leaves()])

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
                lhd = node_lookup["likelihood"]
                other_node = node_lookup["node"]
                lhd = df.loc[seq_id + "_" + reattachment_edge, "likelihood_ratio"]
                other_node = reattachment_node
                break


    # Calculate EDPL for the taxon seq_id:
    edpl = 0
    for n1, node_lookup1 in node_lookup_dictionary.items():
        for n2, node_lookup2 in node_lookup_dictionary.items():
            if n1 != n2:
                edpl += dist(n1, n2, node_lookup1["node"], node_lookup2["node"])*node_lookup1["likelihood"]*node_lookup2["likelihood"]

    # edpl /= likelihood(reduced_tree_file)???
    all_taxa_df[seq_id] = [edpl]

all_taxa_df = pd.DataFrame(all_taxa_df).transpose()
all_taxa_df.columns = ["edpl"]
all_taxa_df.to_csv(output_file)
