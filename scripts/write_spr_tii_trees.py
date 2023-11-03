import pandas as pd
from ete3 import Tree
import subprocess
import os


full_tree_file = snakemake.input.full_tree
data_folder = snakemake.params.data_folder

full_tree = Tree(full_tree_file)
taxon_list = full_tree.get_leaf_names()

for taxon in taxon_list:
    full_tree = Tree(full_tree_file)
    reduced_tree_file = (
        data_folder
        + "reduced_alignments/"
        + taxon
        + "/reduced_alignment.fasta.treefile"
    )
    reduced_tree = Tree(reduced_tree_file)
    node = full_tree.search_nodes(name=taxon)[0]
    node.delete()
    # write the trees for TII/SPR dist calculation into a single text file to be able
    # to use rspr software
    with open(data_folder + "reduced_alignments/" + taxon + "/tii_trees.txt", "w") as f:
        f.write(reduced_tree.write(format=9) + "\n")
        f.write(full_tree.write(format=9))
