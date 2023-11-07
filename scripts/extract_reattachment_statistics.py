from ete3 import Tree
import pandas as pd
import json
import re

epa_result_files = snakemake.input.epa_results
full_tree_file = snakemake.input.full_tree

reattached_trees = snakemake.output.reattached_trees
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
    pattern = re.compile(r"([\w]+):[0-9.]+\{" + str(edge_num) + "\}")
    match = pattern.search(new_newick_str)
    sibling_node_id = match.group(1)

    # delete edge numbers in curly brackets
    ete_newick_str = re.sub(r"\{\d+\}", "", new_newick_str)

    # add new node to tree
    tree = Tree(ete_newick_str, format=2)
    for node in tree.traverse():
        if not node.is_leaf() and not node.is_root():
            node.name = str(int(node.support))
    sibling = tree.search_nodes(name=str(sibling_node_id))[0]
    reattachment_branch_length = sibling.dist

    dist_from_parent = reattachment_branch_length - distal_length
    # support of new internal node shall be 0
    new_internal_node = sibling.up.add_child(name=0, dist=dist_from_parent)
    sibling.detach()
    new_internal_node.add_child(sibling, dist=distal_length)
    new_internal_node.add_child(name=seq_id, dist=pendant_length)

    # add support values back to tree and save reattached tree (new internal node gets support 1.0)
    for node in [node for node in tree.traverse() if not node.is_leaf()]:
        if node.name in int_node_dict:
            node.name = int_node_dict[node.name]
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


full_tree = Tree(full_tree_file)

output = []
for seq_id in seq_ids:
    epa_file = [f for f in epa_result_files if "/" + seq_id + "/" in f][0]
    with open(epa_file, "r") as f:
        dict = json.load(f)

    # compute and safe reattached tree
    tree_file = [f for f in reattached_trees if "/" + seq_id + "/" in f][0]
    placements = dict["placements"][0]["p"]
    # TODO: Check if this is always the best placement and what it means if we have more than one placement in epa_result.jplace
    if isinstance(placements[0], list):
        placements = placements[0]
    edge_num = placements[0]
    likelihood = placements[1]
    like_weight_ratio = placements[2]
    distal_length = placements[3]
    pendant_length = placements[4]

    reattached_tree, reattachment_branch_length = get_reattached_tree(
        dict["tree"], placements[0], seq_id, distal_length, pendant_length
    )
    with open(tree_file, "w") as f:
        f.write(reattached_tree.write())

    rf_distance = full_tree.robinson_foulds(reattached_tree, unrooted_trees=True)[0]
    taxon_height = calculate_taxon_height(reattached_tree, seq_id)
    output.append(
        [
            seq_id,
            likelihood,
            like_weight_ratio,
            reattachment_branch_length,
            pendant_length,
            rf_distance,
            taxon_height,
        ]
    )
df = pd.DataFrame(
    output,
    columns=[
        "seq_id",
        "likelihood",
        "like_weight_ratio",
        "reattachment_branch_length",
        "pendant_branch_length",
        "rf_distance",
        "taxon_height",
    ],
)
df.to_csv(output_csv)
