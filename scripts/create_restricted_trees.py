from ete3 import Tree

full_tree_file = snakemake.input.full_tree
restricted_tree_files = snakemake.output.restricted_trees

full_tree = Tree(full_tree_file)
leaf_names = [filename.split("/")[1] for filename in restricted_tree_files]

for i in range(len(leaf_names)):
    leaf_name = leaf_names[i]
    restricted_tree = full_tree.copy()
    target_leaf = restricted_tree.search_nodes(name=leaf_name)[0]
    target_leaf.delete()
    restricted_tree.write(outfile = restricted_tree_files[i])
    