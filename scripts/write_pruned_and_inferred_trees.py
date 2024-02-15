from ete3 import Tree

full_treefile = snakemake.input.full_tree
inferred_tree_file = snakemake.input.inferred_trees
both_trees_filename = snakemake.output.both_trees
seq_id = snakemake.params.seq_id

subdir = full_treefile.split("/")[1]

full_tree = Tree(full_treefile)
# get pruned tree
pruned_tree = full_tree.copy()
leaf = pruned_tree.search_nodes(name = seq_id)[0]
leaf.delete()
pruned_tree_nwk = pruned_tree.write()
# get inferred tree
with open(inferred_tree_file, "r") as f:
    inferred_tree = f.readline().strip("\n")
# write pruned and inferred tree to one file
with open(both_trees_filename, "w") as f:
    f.write(pruned_tree_nwk + "\n")
    f.write(inferred_tree + "\n")