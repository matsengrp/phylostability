import pandas as pd
from ete3 import Tree

tree_files = snakemake.input.treefiles
ml_files = snakemake.input.mlfiles
mldist_file = snakemake.input.mldist_file
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
    return min(
        [
            taxon_parent.get_distance(leaf)
            for leaf in input_tree.get_leaves()
            if leaf != taxon
        ]
    )


# return the branch length of the reattachment edge
def get_reattachment_branch_length(input_tree_file, taxon_name):
    with open(input_tree_file, "r") as f:
        input_tree = Tree(f.readlines()[0].strip())
    taxon = input_tree & taxon_name
    return taxon.dist


def leaf_distances(node, excluded_leaves=[]):
    return [
        (node.get_distance(l), l.name)
        for l in node.get_leaves()
        if l not in excluded_leaves
    ]


# return the distance between the closest leaf on one side of the branch to the closest leaf on the other
def get_reattachment_closest_leaf_distances(input_tree_file, taxon_name):
    with open(input_tree_file, "r") as f:
        input_tree = Tree(f.readlines()[0].strip())
    distances = pd.read_table(
        mldist_file, skiprows=[0], header=None, delim_whitespace=True, index_col=0
    )
    distances.columns = distances.index
    taxon = input_tree & taxon_name
    edge_child = taxon.get_sisters()[0]
    edge_parent = taxon.get_sisters()[-1] if taxon.up.is_root() else taxon.up.up
    leaf1 = edge_child.get_closest_leaf()[0].name
    leaf2_distances = leaf_distances(
        edge_parent, excluded_leaves=taxon.get_leaves() + edge_child.get_leaves()
    )
    leaf2 = min(leaf2_distances)[1]
    return [distances.loc[[leaf2], [leaf1]].sum().sum(), leaf1, leaf2]


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
    reattachment_branch_nodedata = get_reattachment_closest_leaf_distances(
        tree_file, seq_id
    )
    df[seq_id + "_" + str(i + 1)] = [
        branchlengths,
        taxon_height,
        likelihood,
        rf_distance,
        reattachment_branch_length,
        reattachment_branch_nodedata[0],
        reattachment_branch_nodedata[1],
        reattachment_branch_nodedata[2],
    ]

df = pd.DataFrame(df).transpose()
df.columns = [
    "branchlengths",
    "taxon_height",
    "likelihood",
    "rf_distance",
    "reattachment_branch_length",
    "reattachment_branch_distances",
    "reattachment_child",
    "reattachment_parent",
]
seq_id = df.index.to_list()[0]
suffix = seq_id.split("_")[-1]
seq_id = seq_id[: -len(suffix) - 1]
df["seq_id"] = seq_id
df["likelihood_ratio"] = df.likelihood / (
    df.likelihood.sum() if df.likelihood.sum() != 0 else 1
)
df.to_csv(df_name)
