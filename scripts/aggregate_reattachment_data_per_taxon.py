import pandas as pd
from ete3 import Tree

tree_files = snakemake.input.treefiles
reduced_tree_file = snakemake.input.reduced_treefile
ml_file = snakemake.input.ml_files
output_file = snakemake.output.output_file
seq_id = snakemake.params.seq_id
edge_ids = snakemake.params.edges
df = snakemake.params.global_dictionary


# load tree on full taxon set
with open(full_tree_file, "r") as f:
    main_tree = Tree(f.readlines()[0])

# create a lookup dictionary between the main tree's nodes and each of the nodes in each of the tree_files
node_lookup_dictionary = {n:[] for n in main_tree.traverse("postorder")}

# load all of the reattached-taxon trees for the specific seq_id
other_trees = []
for tree_file in tree_files:
    with open(tree_file, "r") as f:
        other_tree = Tree(f.readlines()[0])
    # save tree to list
    other_trees.append(other_tree)
    # find the correspondence between this tree's nodes and the nodes in the main_tree
    for node, node_lookup in node_lookup_dictionary.items():
        if node.is_leaf():
            leafname = node.name
            node_lookup.append(other_tree & leafname)
        else:
            node_leafset = set([l.name for l in node.get_leaves()])
            for other_tree_node in other_tree.traverse("postorder"):
                if other_tree_node.name != seq_id: # remember that taxon 'seq_id' has been pruned from main_tree
                    other_tree_node_leafset = set([l.name for l in other_tree_node.get_leaves() if l.name != seq_id])
                    if other_tree_node_leafset == node_leafset:
                        node_lookup.append(other_tree_node)
                        break

likelihoods = []
