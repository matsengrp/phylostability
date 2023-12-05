from ete3 import Tree
import json

from utils import *

epa_result = snakemake.input.epa_result
restricted_tree_file = snakemake.input.restricted_trees

reattached_tree_file = snakemake.output.reattached_trees

with open(epa_result, "r") as f:
    dict = json.load(f)

seq_id = reattached_tree_file.split("/")[-2]
restricted_tree = Tree(restricted_tree_file)

placements = dict["placements"][0][
    "p"
]  # this is a list of lists, each containing information for one reattachment
newick_trees = []
num_likely_reattachments = len(placements)
# get values for which we need to iterate through all best reattachments
reattached_trees = []
for placement in placements:
    reattached_tree = get_reattached_tree(
        dict["tree"], placement[0], seq_id, placement[3], placement[4]
    )[0]
    reattached_trees.append(reattached_tree)
newick_trees.append(reattached_tree.write(format=0))

with open(reattached_tree_file, "w") as f:
    for newick_str in newick_trees:
        f.write(newick_str + "\n")
