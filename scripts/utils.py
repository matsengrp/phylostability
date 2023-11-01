import pandas as pd
from ete3 import Tree
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from io import StringIO


def ete_dist(node1, node2, topology_only=False):
    # if one of the nodes is a leaf and child of the other one, we need to add one
    # to their distance because get_distance() returns number of nodes between
    # given nodes. E.g. if node1 and node2 are connected by edge, this would be
    # distance 0, but it should be 1
    add_to_dist = 0
    if node2 in node1.get_ancestors():
        leaf = node1.get_leaves()[0]
        if node1 == leaf and topology_only == True:
            add_to_dist = 1
        return (
            node2.get_distance(leaf, topology_only=topology_only)
            - node1.get_distance(leaf, topology_only=topology_only)
            + add_to_dist
        )
    else:
        leaf = node2.get_leaves()[0]
        if node2 == leaf and topology_only == True and node1 in node2.get_ancestors():
            add_to_dist = 1
        return (
            node1.get_distance(leaf, topology_only=topology_only)
            - node2.get_distance(leaf, topology_only=topology_only)
            + add_to_dist
        )


def aggregate_taxon_edge_dfs(csv_list):
    """
    Aggregate all dataframes in csv_list into one dataframe for plotting.
    """
    dfs = []
    for csv_file in csv_list:
        taxon_df = pd.read_csv(csv_file)
        dfs.append(taxon_df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={0: "seq_id"})
    return df


def aggregate_and_filter_by_likelihood(taxon_edge_csv_list, p, hard_threshold=3):
    """
    Reads and aggregates taxon_edge_csv_list dataframes for all taxa, while
    also filtering out by likelihood.
    p is a value between 0 and 1, so that only taxon reattachments whose trees have
    likelihood greater than max_likelihood - p * (max_likelihood - min_likelihood)
    are added to the aggregated dataframe.
    """
    dfs = []
    for csv_file in taxon_edge_csv_list:
        taxon_df = pd.read_csv(csv_file)
        # filter by likelihood
        min_likelihood = taxon_df["likelihood"].min()
        max_likelihood = taxon_df["likelihood"].max()
        threshold = max_likelihood - p * (max_likelihood - min_likelihood)
        filtered_df = taxon_df[taxon_df["likelihood"] >= threshold]
        if len(filtered_df) > hard_threshold:
            filtered_df = filtered_df.nlargest(hard_threshold, "likelihood")
        # append to df for all taxa
        dfs.append(filtered_df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={0: "seq_id"})
    if "Unnamed: 0.1" in df.columns:
        df.set_index("Unnamed: 0.1", inplace=True)
    elif "Unnamed: 0" in df.columns:
        df.set_index("Unnamed: 0", inplace=True)
    return df


def get_ml_dist(mldist_file):
    """
    Read ml_dist from input filename and return df with rownames=colnames=seq_ids.
    """
    ml_distances = pd.read_table(
        mldist_file,
        skiprows=[0],
        header=None,
        delim_whitespace=True,
        index_col=0,
    )
    ml_distances.columns = ml_distances.index
    return ml_distances


def get_best_reattached_tree(seq_id, all_taxon_edge_df, data_folder):
    """
    Return best reattached tree with seq_id (best means highest likelihood)
    of reattachment edge as given in all_taxon_edge_df.
    """
    filtered_df = all_taxon_edge_df.loc[all_taxon_edge_df["seq_id"] == seq_id]
    if isinstance(filtered_df.index[0], str):
        best_edge_id = filtered_df["likelihood"].idxmax()
    else:
        best_edge_id = filtered_df.loc[filtered_df["likelihood"].idxmax()][0]
    if not isinstance(best_edge_id, str):
        best_edge_id = filtered_df["likelihood"].idxmax()
    best_edge_id = best_edge_id.split("_")[-1]
    tree_filepath = (
        data_folder
        + "reduced_alignments/"
        + seq_id
        + "/reduced_alignment.fasta_add_at_edge_"
        + str(best_edge_id)
        + ".nwk_branch_length.treefile"
    )
    tree = Tree(tree_filepath)
    return tree


def get_best_reattached_tree_distances_to_seq_id(
    seq_id, all_taxon_edge_df, data_folder
):
    """ "
    Return dictionary with distances in tree with highest likelihood
    among all reattached trees (as per all_taxon_edge_df).
    """
    tree = get_best_reattached_tree(seq_id, all_taxon_edge_df, data_folder)
    leaves = tree.get_leaf_names()
    # Compute distance for each leaf to seq_id leaf
    distances = {}
    for i, leaf in enumerate(leaves):
        if leaf != seq_id:
            distances[leaf] = tree.get_distance(leaf, seq_id)
    return distances


def get_closest_msa_sequences(seq_id, mldist_file, p):
    """
    Returns list of names of p closest sequences in MSA to seq_id.
    """
    ml_distances = get_ml_dist(mldist_file)
    seq_id_row = ml_distances[seq_id]
    seq_id_row = seq_id_row.drop(seq_id)
    top_p = seq_id_row.nsmallest(p)
    row_names = top_p.index.tolist()
    return row_names


def get_seq_dists_to_seq_id(seq_id, mldist_file, no_seqs=None):
    """
    Returns dict of no_seqs closest distances in MSA to seq_id,
    containing names as keys and distances as values.
    """
    ml_distances = get_ml_dist(mldist_file)
    d = {}
    for seq in ml_distances.index:
        if seq != seq_id:
            d[seq] = ml_distances[seq_id][seq]
    # only look at closest no_seqs sequences to seq_id
    if no_seqs != None:
        top_items = sorted(d.items(), key=lambda x: x[1], reverse=False)[:no_seqs]
        top_dict = dict(top_items)
        return top_dict
    return d


def get_nodes_on_path(tree, node1, node2):
    """
    For any two input nodes, returns a list of all nodes on the
    path between these nodes in the tree.
    Input nodes can be nodes or node names in the tree.
    """
    if isinstance(node1, str):
        node1 = tree.search_nodes(name=node1)[0]
    if isinstance(node2, str):
        node2 = tree.search_nodes(name=node2)[0]
    mrca = tree.get_common_ancestor(node1, node2)
    nodes_on_path = []
    for node in [node1, node2]:
        ancestor = node
        nodes_on_path.append(ancestor)
        while ancestor != mrca:
            ancestor = ancestor.up
            if ancestor not in nodes_on_path:
                nodes_on_path.append(ancestor)
    return nodes_on_path


def compute_nj_tree(d):
    """
    Compute NJ tree for distance matrix d (pd.DataFrame).
    Returns a tree in ete format
    """
    # convert d into lower triangular distance matrix as list of lists
    matrix = []
    i = 1
    for row in d.index:
        matrix.append([l for l in d[row][:i]])
        i += 1
    distance_matrix = DistanceMatrix(names=d.index.to_list(), matrix=matrix)
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(distance_matrix)
    # convert tree to ete
    tree_newick = StringIO()
    Phylo.write(tree, tree_newick, "newick")
    tree_newick = tree_newick.getvalue()
    tree = Tree(tree_newick, format=1)
    return tree
