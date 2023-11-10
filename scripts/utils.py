import pandas as pd
from ete3 import Tree
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from io import StringIO


def get_seq_id_file(seq_id, files):
    return [f for f in files if "/" + seq_id + "/" in f][0]


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


def get_reattached_trees(treefile, best=True):
    with open(treefile, "r") as f:
        content = f.readlines()
        if best:
            return Tree(content[0].strip())
        else:
            return [Tree(str) for str in content.strip()]


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
