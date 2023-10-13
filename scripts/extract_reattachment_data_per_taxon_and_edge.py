import pandas as pd
from ete3 import Tree

tree_files = snakemake.input.treefiles
ml_files = snakemake.input.mlfiles
full_tree_file = snakemake.input.full_tree_file
df_name = snakemake.output.csv_name
seq_id = snakemake.params.seq_id


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
            lengths[str(ctr)] = [node_str, node.dist]
        ctr += 1
    return lengths


def get_taxon_likelihood(input_file):
    likelihood = 0
    ll_str = "Log-likelihood"
    with open(input_file, "r") as f:
        for line in f.readlines():
            if ll_str in line:
                likelihood = float(line.split(": ")[-1].split(" ")[0].strip())
                break
    return likelihood


# return the distance to the closest leaf of the taxon specified
def calculate_taxon_height(input_tree_file, taxon_name):
    with open(input_tree_file, "r") as f:
        tree_nwk = f.readlines()[0].strip()
    input_tree = Tree(tree_nwk)
    taxon = input_tree & taxon_name
    taxon_parent = taxon.up
    input_tree.delete(taxon)
    return taxon_parent.get_closest_leaf()[1]

# return the branch length of the reattachment edge
def get_reattachment_branch_length(input_tree_file, taxon_name):
    with open(input_tree_file, "r") as f:
        input_tree = Tree(f.readlines()[0].strip())
    taxon = input_tree & taxon_name
    return taxon.dist

def get_distance_to_full_tree(reattached_tree_file, full_tree_file):
    with open(reattached_tree_file, "r") as f:
        reattached_tree_nwk = f.readlines()[0].strip()
    reattached_tree = Tree(reattached_tree_nwk)
    with open(full_tree_file, "r") as f:
        full_tree_nwk = f.readlines()[0].strip()
    full_tree = Tree(full_tree_nwk)
    return reattached_tree.robinson_foulds(full_tree, unrooted_trees=True)[0]


df = {}
# df is actually a dictionary whose values are lists, each list should be a row in the dataframe
for i, tree_file in enumerate(tree_files):
    ml_file = ml_files[i]
    branchlengths = get_branch_lengths(tree_file)
    taxon_height = calculate_taxon_height(tree_file, seq_id)
    likelihood = get_taxon_likelihood(ml_file)
    rf_distance = get_distance_to_full_tree(tree_file, full_tree_file)
    reattachment_branch_length = get_reattachment_branch_length(tree_file, seq_id)
    df[seq_id + "_" + str(i + 1)] = [
        branchlengths,
        taxon_height,
        likelihood,
        rf_distance,
        reattachment_branch_length,
    ]

df = pd.DataFrame(df).transpose()
df.columns = ["branchlengths", "taxon_height", "likelihood", "rf_distance", "reattachment_branch_length"]
seq_id = df.index.to_list()[0]
suffix = seq_id.split("_")[-1]
seq_id = seq_id[:-len(suffix)-1]
df["seq_id"] = seq_id
df["likelihood_ratio"] = df.likelihood / (
    df.likelihood.sum() if df.likelihood.sum() != 0 else 1
)
df.to_csv(df_name)
