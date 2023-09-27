from ete3 import Tree

tree_file = snakemake.input[1]
full_tree_file = snakemake.input[2]
ml_file = snakemake.input[3]
seq_id = snakemake.params.seq_id
edge_id = snakemake.params.edge
df = snakemake.params.global_dictionary


# extract a dictionary of branch lengths keyed on the child node name of the corresponding branch
def get_branch_lengths(input_tree_file):
    with open(input_tree_file, "r") as f:
        tree_nwk = f.readlines()[0].strip()
    input_tree = Tree(tree_nwk)
    lengths = {}
    ctr = 0
    for node in input_tree.traverse("postorder"):
        if len(node.get_ancestors()) > 0:
            node_str = ",".join(sorted(list([l.name for l in node.get_leaves()])))
            lengths[str(ctr)] = [node_str, node.get_ancestors()[0].get_distance(node)]
        ctr += 1
    return lengths


def get_taxon_likelihood(input_file):
    likelihood = 0
    ll_str = "Log-likelihood of the tree: "
    with open(input_file, "r") as f:
        for line in f.readlines()[0]:
            if ll_str in line:
                likelihood = line.split(ll_str)[-1].split(" ")[0]
                break
    return likelihood


# return the distance to the closest leaf of the taxon specified
def calculate_taxon_height(input_tree_file, taxon_name):
    with open(input_tree_file, "r") as f:
        tree_nwk = f.readlines()[0].strip()
    input_tree = Tree(tree_nwk)
    taxon = input_tree & taxon_name
    taxon_parent = taxon.get_ancestors()[0]
    input_tree.delete(taxon)
    return taxon_parent.get_closest_leaf()[1]


def get_distance_to_full_tree(reattached_tree_file, full_tree_file):
    with open(reattached_tree_file, "r") as f:
        reattached_tree_nwk = f.readlines()[0].strip()
    reattached_tree = Tree(reattached_tree_nwk)
    with open(full_tree_file, "r") as f:
        full_tree_nwk = f.readlines()[0].strip()
    full_tree = Tree(full_tree_nwk)
    return reattached_tree.robinson_foulds(full_tree, unrooted_trees=True)[0]


branchlengths = get_branch_lengths(tree_file)
taxon_height = calculate_taxon_height(tree_file, seq_id)
likelihood = get_taxon_likelihood(ml_file)
rf_distance = get_distance_to_full_tree(tree_file, full_tree_file)

# df is actually a dictionary whose values are lists, each list should be a row in the dataframe
df[seq_id+"_"+edge_id] = [branchlengths, taxon_height, likelihood, rf_distance]
