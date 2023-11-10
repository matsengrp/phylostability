import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import re
import pickle
import math
import itertools

from utils import *


def reattachment_distance_to_low_support_node(
    sorted_taxon_tii_list, reattached_tree_files, bootstrap_threshold=1
):
    """
    Plot (topological) distance of reattachment position in best_reattached_tree to
    nearest low bootstrap node for each seq_id.
    """
    df = []
    for seq_id, tii in sorted_taxon_tii_list:
        reattached_tree_file = get_seq_id_file(seq_id, reattached_tree_files)
        reattached_tree = get_reattached_trees(reattached_tree_file)
        reattachment_node = reattached_tree.search_nodes(name=seq_id)[0].up
        all_bootstraps = [
            node.support
            for node in reattached_tree.traverse()
            if not node.is_root() and not node.is_leaf() and node != reattachment_node
        ]
        q = np.quantile(all_bootstraps, bootstrap_threshold)
        # parent of seq_id is reattachment_node
        min_dist_found = float("inf")
        for node in [
            node
            for node in reattached_tree.traverse()
            if not node.is_leaf() and node != reattachment_node
        ]:
            # take max of distance of two endpoints of nodes at which we reattach
            if not node.is_root():
                dist = max(
                    [
                        ete_dist(node, reattachment_node, topology_only=True),
                        ete_dist(node.up, reattachment_node, topology_only=True),
                    ]
                )
            else:
                dist = ete_dist(node, reattachment_node, topology_only=True)
            if node.support < q and dist < min_dist_found:
                min_dist_found = dist
        df.append([seq_id + " " + str(tii), min_dist_found])
    df = pd.DataFrame(df, columns=["seq_id", "distance"])
    sns.scatterplot(data=df, x="seq_id", y="distance")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


def nj_tii(
    mldist_file,
    sorted_taxon_tii_list,
    reduced_mldist_files,
    plot_filepath,
    ratio_plot_filepath,
):
    """
    Compute Neighbour Joining TII and compare to ML TII as well as average ratio of NJ
    tree distances to sequence distances for each reduced tree as a measure of how tree
    like the smaller alignment is.
    """

    full_mldist = get_ml_dist(mldist_file)
    full_tree = compute_nj_tree(full_mldist)
    # Convert tree from biopython to ete format

    df = []
    tree_likeness = []
    for seq_id, tii in sorted_taxon_tii_list:
        f = get_seq_id_file(seq_id, reduced_mldist_files)
        reduced_mldist = get_ml_dist(f)
        reduced_tree = compute_nj_tree(reduced_mldist)

        leaves = reduced_tree.get_leaf_names()
        # compute ratio of nj tree distances and sequence distances
        tl = 0
        for leaf1, leaf2 in itertools.combinations(leaves, 2):
            reduced_tree_dist = reduced_tree.get_distance(leaf1, leaf2)
            reduced_ml_dist = reduced_mldist[leaf1][leaf2]
            tl += reduced_tree_dist / reduced_ml_dist
        tree_likeness.append(
            [
                seq_id,
                leaf1,
                leaf2,
                tl
                / (math.factorial(len(leaves)) / math.factorial(len(leaves) - 2) / 2),
            ]
        )

        rf_dist = full_tree.robinson_foulds(reduced_tree, unrooted_trees=True)[0]
        df.append([seq_id + " " + str(tii), rf_dist])
    df = pd.DataFrame(df, columns=["seq_id", "rf_distance"])
    sns.scatterplot(data=df, x="seq_id", y="rf_distance")
    plt.xticks(
        rotation=90,
    )
    plt.title("NJ TII vs ML TII")
    plt.xlabel("seq_id")
    plt.ylabel("NJ TII")
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()

    tree_likeness_df = pd.DataFrame(
        tree_likeness, columns=["seq_id", "leaf1", "leaf2", "ratio"]
    )
    sns.scatterplot(data=tree_likeness_df, x="seq_id", y="ratio")
    plt.xticks(
        rotation=90,
    )
    plt.title("NJ distance : sequence distance ratio")
    plt.xlabel("seq_id")
    plt.ylabel("Ratio")
    plt.tight_layout()
    plt.savefig(ratio_plot_filepath)
    plt.clf()


def order_of_distances_to_seq_id(
    sorted_taxon_tii_list,
    mldist_file,
    reattached_tree_files,
    plot_filepath,
    barplot_filepath,
):
    """
    Plot difference in tree and sequence distance for sequences of taxon1 and taxon 2
    where distance of taxon1 to seq_id is smaller than taxon2 to seq_id in best reattached
    tree, but greater in terms of sequence distance, for each possible seq_id.
    """
    mldist = get_ml_dist(mldist_file)
    max_mldist = mldist.max().max()
    df = []
    hist_counts = []
    for seq_id, tii in sorted_taxon_tii_list:
        reattached_tree_file = get_seq_id_file(seq_id, reattached_tree_files)
        reattached_tree = get_reattached_trees(reattached_tree_file)
        # find maximum distance between any two leaves in best_reattached_tree
        max_tree_dist = 0
        for leaf1, leaf2 in itertools.combinations(reattached_tree.get_leaves(), 2):
            if leaf1.get_distance(leaf2) > max_tree_dist:
                max_tree_dist = leaf1.get_distance(leaf2)

        leaves = [leaf for leaf in reattached_tree.get_leaf_names() if leaf != seq_id]
        reattachment_node = reattached_tree.search_nodes(name=seq_id)[0].up
        all_bootstraps = [
            node.support
            for node in reattached_tree.traverse()
            if not node.is_leaf() and not node.is_root() and node != reattachment_node
        ]
        q = np.quantile(all_bootstraps, 0.1)
        subset = []
        for leaf1, leaf2 in itertools.combinations(leaves, 2):
            mrca = reattached_tree.get_common_ancestor([leaf1, leaf2])
            if mrca.support >= q or mrca.support == 1.0:
                continue

            # # # Careful: Restricting by leaves being "on the same side" seems biased by rooting
            # # # we only consider leaves on the same side of seq_id in the tree
            # mrca = reattached_tree.get_common_ancestor([leaf1, leaf2])
            # p1 = get_nodes_on_path(reattached_tree, seq_id, leaf1)
            # p2 = get_nodes_on_path(reattached_tree, seq_id, leaf2)
            # if mrca not in p1 or mrca not in p2:
            #     continue

            # # only consider leaf1 and leaf2 if their path in reduced tree contains
            # # mostly low bootstrap nodes
            # low_support = high_support = 0
            # p = get_nodes_on_path(reduced_tree, leaf1, leaf2)
            # for node in p:
            #     if node.support < q:
            #         low_support += 1
            #     else:
            #         high_support += 1
            # if not high_support <= 4 * low_support:
            #     continue
            # add differences between sequence and tree distance to df if order of
            # distances between leaf1 and leaf2 to seq_id are different in tree and
            # msa distance matrix
            tree_dist_leaf1 = (
                reattached_tree.get_distance(seq_id, leaf1, topology_only=True)
                / max_tree_dist
            )
            tree_dist_leaf2 = (
                reattached_tree.get_distance(seq_id, leaf2, topology_only=True)
                / max_tree_dist
            )
            seq_dist_leaf1 = mldist[seq_id][leaf1] / max_mldist
            seq_dist_leaf2 = mldist[seq_id][leaf2] / max_mldist
            if (
                tree_dist_leaf1 / tree_dist_leaf2 < 1
                and seq_dist_leaf1 / seq_dist_leaf2 > 1
            ):
                difference = (
                    seq_dist_leaf1 / seq_dist_leaf2 - tree_dist_leaf1 / tree_dist_leaf2
                )
                subset.append([seq_id + " " + str(tii), leaf1, leaf2, difference])
            elif (
                tree_dist_leaf2 / tree_dist_leaf1 < 1
                and seq_dist_leaf2 / seq_dist_leaf1 > 1
            ):
                difference = (
                    seq_dist_leaf2 / seq_dist_leaf1 - tree_dist_leaf2 / tree_dist_leaf1
                )
                subset.append([seq_id + " " + str(tii), leaf2, leaf1, difference])
        subset_df = pd.DataFrame(
            subset, columns=["seq_id", "leaf1", "leaf2", "difference"]
        )
        # # Select the 10 rows with highest "difference"
        # sorted_list = sorted(subset, key=lambda x: x[3], reverse=True)
        # subset = sorted_list[:5]

        for other_seq in [s for s, tii in sorted_taxon_tii_list if s != seq_id]:
            count = subset_df["leaf1"].value_counts().get(other_seq, 0) + subset_df[
                "leaf2"
            ].value_counts().get(other_seq, 0)
            hist_counts.append([seq_id + " " + str(tii), other_seq, count])
        df += subset
    df = pd.DataFrame(df, columns=["seq_id", "leaf1", "leaf2", "difference"])

    plt.figure(figsize=(10, 6))
    sns.stripplot(data=df, x="seq_id", y="difference")
    plt.xticks(
        rotation=90,
    )
    plt.title("Difference between sequence distance and tree distance ratio")
    plt.ylabel("Difference in distance ratios")
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()

    # Count the occurrences of each 'seq_id' in df
    counts = df["seq_id"].value_counts().to_frame()
    tii_list = []
    for seq_id in counts.index:
        tii_list.append(float(seq_id.split(" ")[1]))
    counts["tii"] = tii_list
    counts = counts.sort_values(by="tii")
    counts.columns = ["count", "tii"]

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=counts, x=counts.index, y="count")
    plt.xticks(
        rotation=90,
    )
    plt.ylabel("count")
    plt.title("Number of change in order of distances (tree vs msa) to seq_id")
    plt.tight_layout()
    plt.savefig(barplot_filepath)
    plt.clf()


def seq_and_tree_dist_diff(
    sorted_taxon_tii_list,
    mldist_file,
    reattached_tree_files,
    ratio_plot_filepath,
):
    """
    Plot difference and ratio of distance between seq_id and other sequences in alignment
    to corresponding distance in best reattached tree.
    """
    ml_distances = get_ml_dist(mldist_file)
    distance_ratios = []
    for seq_id, tii in sorted_taxon_tii_list:
        treefile = get_seq_id_file(seq_id, reattached_tree_files)
        tree = get_reattached_trees(treefile)
        leaves = [leaf for leaf in tree.get_leaf_names() if leaf != seq_id]
        # fill distance_ratios
        for leaf in leaves:
            ratio = ml_distances[seq_id][leaf] / tree.get_distance(seq_id, leaf)
            distance_ratios.append([seq_id + " " + str(tii), ratio])

    plt.figure(figsize=(10, 6))  # Adjust figure size if needed
    df = pd.DataFrame(distance_ratios, columns=["seq_id", "distance_ratio"])
    sns.stripplot(data=df, x="seq_id", y="distance_ratio")
    # plt.axhline(1, color="red")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(ratio_plot_filepath)
    plt.clf()


def get_reattachment_distances(reduced_tree, reattachment_trees_file, seq_id):
    """
    Return dataframe with distances between reattachment locations of trees in
    file reattachment_trees_file
    """
    with open(reattachment_trees_file, "r") as f:
        content = f.readlines()
    trees = [Tree(nwk_tree.strip()) for nwk_tree in content]
    if len(trees) == 1:
        return [0]
    reattachment_node_list = []
    for tree in trees:
        # cluster is set of leaves below lower node of reattachment edge
        cluster = tree.search_nodes(name=seq_id)[0].up.get_leaf_names()
        cluster.remove(seq_id)
        if len(cluster) == 1:  # reattachment above leaf
            reattachment_node = tree.search_nodes(name=cluster[0])[0]
        else:
            mrca = reduced_tree.get_common_ancestor(cluster)
            reattachment_node = mrca
            reattachment_node_list.append(reattachment_node)
    reattachment_distances = []
    for node1, node2 in itertools.combinations(reattachment_node_list, 2):
        reattachment_distances.append(ete_dist(node1, node2, topology_only=True))
    return reattachment_distances


def plot_reattachment_distances(
    sorted_taxon_tii_list, reduced_tree_files, reattached_tree_files, plot_filepath
):
    """
    Plot distances between best reattachment locations returned by epa-ng.
    We plot distance 0 if there is only one best reattachment location (i.e. LWR > 0.99)
    """
    df = []
    for seq_id, tii in sorted_taxon_tii_list:
        reduced_tree_file = [
            file for file in reduced_tree_files if "/" + seq_id + "/" in file
        ][0]
        treefile = [
            file for file in reattached_tree_files if "/" + seq_id + "/" in file
        ][0]
        reduced_tree = Tree(reduced_tree_file)
        reattachment_distances = get_reattachment_distances(
            reduced_tree, treefile, seq_id
        )
        df += [[seq_id + " " + str(tii), d] for d in reattachment_distances]
    df = pd.DataFrame(df, columns=["seq_id", "distances"])
    sns.stripplot(data=df, x="seq_id", y="distances")
    plt.title("Distances between best reattachment positions")
    plt.ylabel("Topological distances")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


# def dist_of_likely_reattachments(
#     sorted_taxon_tii_list, all_taxon_edge_df, reattachment_distance_csv, filepath
# ):
#     """
#     Plot distance between all pairs of reattachment locations in all_taxon_edge_df.
#     We plot these distances for each taxon, sorted according to increasing TII, and
#     colour the datapooints by log likelihood difference.
#     """
#     pairwise_df = []
#     # create DataFrame containing all distances and likelihood
#     for i, (seq_id, tii) in enumerate(sorted_taxon_tii_list):
#         reattachment_distances = get_reattachment_distances(
#             all_taxon_edge_df, reattachment_distance_csv, seq_id
#         )
#         max_likelihood_reattachment = reattachment_distances.loc[
#             reattachment_distances["likelihoods"].idxmax()
#         ].name
#         # create new df with pairwise distances + likelihood difference
#         for i in range(len(reattachment_distances)):
#             for j in range(i + 1, len(reattachment_distances)):
#                 best_reattachment = False
#                 if (
#                     reattachment_distances.columns[i] == max_likelihood_reattachment
#                 ) or (reattachment_distances.columns[j] == max_likelihood_reattachment):
#                     best_reattachment = True
#                 ll_diff = abs(
#                     reattachment_distances["likelihoods"][i]
#                     - reattachment_distances["likelihoods"][j]
#                 )
#                 pairwise_df.append(
#                     [
#                         seq_id + " " + str(tii),
#                         reattachment_distances.columns[i],
#                         reattachment_distances.columns[j],
#                         reattachment_distances.iloc[i, j],
#                         ll_diff,
#                         best_reattachment,
#                     ]
#                 )
#     pairwise_df = pd.DataFrame(
#         pairwise_df,
#         columns=[
#             "seq_id",
#             "edge1",
#             "edge2",
#             "distance",
#             "ll_diff",
#             "best_reattachment",
#         ],
#     )
#     # Filter the DataFrame for each marker type -- cross for distances to best
#     # reattachment position
#     df_marker_o = pairwise_df[pairwise_df["best_reattachment"] == False]
#     df_marker_X = pairwise_df[pairwise_df["best_reattachment"] == True]

#     # Create the plot
#     fig, ax = plt.subplots(figsize=(10, 6))
#     if len(pairwise_df) == 0:
#         print("All taxa have unique reattachment locations.")
#         plt.savefig(filepath)  # save empty plot to not break snakemake output
#         plt.clf()
#         return 0

#     # Plot marker 'o' (only if more than two datapoints)
#     if len(df_marker_o) > 0:
#         sns.stripplot(
#             data=df_marker_o,
#             x="seq_id",
#             y="distance",
#             hue="ll_diff",
#             palette="viridis",
#             alpha=0.7,
#             marker="o",
#             jitter=True,
#             size=7,
#         )

#     # Plot marker 'X'
#     sns.stripplot(
#         data=df_marker_X,
#         x="seq_id",
#         y="distance",
#         hue="ll_diff",
#         palette="viridis",
#         alpha=0.7,
#         marker="X",
#         jitter=True,
#         size=9,
#     )

#     # Add colorbar
#     norm = plt.Normalize(pairwise_df["ll_diff"].min(), pairwise_df["ll_diff"].max())
#     sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=ax, pad=0.1)
#     cbar.set_label("ll_diff")

#     # Other plot customizations
#     plt.legend([], [], frameon=False)
#     plt.ylabel("distance between reattachment locations")
#     plt.title("Distance between optimal reattachment locations")
#     plt.xticks(rotation=90)
#     plt.tight_layout()
#     plt.savefig(filepath)
#     plt.clf()


def get_bootstrap_and_bts_scores(
    reduced_tree_files,
    full_tree_file,
    sorted_taxon_tii_list,
    csv,
    test=False,
):
    """
    Returns three DataFrames:
    (i) branch_scores_df containing bts values for all edges in tree in full_tree_file
    (ii) bootstrap_df: bootstrap support by seq_id for reduced trees (without seq_id)
    (iii) full_tree_bootstrap_df: bootstrap support for tree in full_tree_file
    If test==True, we save the pickled bts_score DataFrame in "test_data/bts_df.p"
    to then be able to use it for testing.
    """
    csv = pd.read_csv(csv)

    bootstrap_df = []

    full_tree = Tree(full_tree_file)
    num_leaves = len(full_tree)

    # initialise branch score dict with sets of leaves representing edges of
    # full_tree
    branch_scores = {}
    all_taxa = full_tree.get_leaf_names()
    full_tree_bootstrap = {}
    for node in full_tree.traverse("postorder"):
        if not node.is_leaf() and not node.is_root():
            sorted_leaves = sorted(node.get_leaf_names())
            leaf_str = ",".join(sorted_leaves)
            full_tree_bootstrap[leaf_str] = node.support
            # ignore pendant edges
            if len(sorted_leaves) < len(all_taxa) - 1:
                s = 0
                for child in node.children:
                    if child.is_leaf():
                        s += 1
                # add 0 for branch score
                branch_scores[",".join(sorted_leaves)] = 0
    full_tree_bootstrap_df = pd.DataFrame(
        full_tree_bootstrap, index=["bootstrap_support"]
    ).transpose()

    for treefile in reduced_tree_files:
        tree = get_reattached_trees(treefile)
        seq_id = treefile.split("/")[-2]
        tii = [p[1] for p in sorted_taxon_tii_list if p[0] == seq_id][0]

        # collect bootstrap support and bts for all nodes in tree
        for node in tree.traverse("postorder"):
            if not node.is_leaf() and not node.is_root():
                # add bootstrap support values for seq_id
                bootstrap_df.append([seq_id, node.support, tii])
                # one edge in the full tree could correspond to two edges in the
                # reduced tree (leaf_str and leaf_str_extended below), if the pruned
                # taxon is attached on that edge. We hence need to add one for each of
                # those, if they are in branch_scores.
                edge_found = False
                leaf_list = node.get_leaf_names()
                leaf_str = ",".join(sorted(leaf_list))
                if leaf_str in branch_scores:
                    branch_scores[leaf_str] += 1
                    continue
                leaf_str_extended = ",".join(sorted(leaf_list + [seq_id]))
                if not edge_found and leaf_str_extended in branch_scores:
                    branch_scores[leaf_str_extended] += 1
                    continue
                # edge ID might be complement of leaf set
                # this could happen if rooting of tree is different to that of full_tree
                complement_leaves = list(
                    set(all_taxa) - set(node.get_leaf_names()) - set(seq_id)
                )
                # ignore node if it represents leaf
                if len(complement_leaves) == 1:
                    continue
                leaf_str = ",".join(complement_leaves)
                if leaf_str in branch_scores:
                    branch_scores[leaf_str] += 1
                    continue
                leaf_str_extended = ",".join(complement_leaves + [seq_id])
                if leaf_str_extended in branch_scores:
                    branch_scores[leaf_str_extended] += 1
    # normalise BTS
    # Divide by num_leaves - 2 if edge is incident to cherry, as it cannot be present in
    # either of the two trees where one of those cherry leaves is pruned
    for branch_score in branch_scores:
        if branch_score.count(",") == 1 or branch_score.count(",") == num_leaves - 3:
            branch_scores[branch_score] *= 100 / (num_leaves - 2)  # normalise bts
        else:
            branch_scores[branch_score] *= 100 / num_leaves
        branch_scores[branch_score] = int(branch_scores[branch_score])
    branch_scores_df = pd.DataFrame(branch_scores, index=["bts"]).transpose()

    bootstrap_df = pd.DataFrame(
        bootstrap_df, columns=["seq_id", "bootstrap_support", "tii"]
    )
    bootstrap_df = bootstrap_df.sort_values("tii")

    if test == True:
        with open("test_data/bts_df.p", "wb") as f:
            pickle.dump(branch_scores_df, file=f)
    return branch_scores_df, bootstrap_df, full_tree_bootstrap_df


def bootstrap_and_bts_plot(
    reduced_tree_files,
    full_tree_file,
    sorted_taxon_tii_list,
    bts_vs_bootstrap_path,
    csv,
):
    (
        branch_scores_df,
        bootstrap_df,
        full_tree_bootstrap_df,
    ) = get_bootstrap_and_bts_scores(
        reduced_tree_files,
        full_tree_file,
        sorted_taxon_tii_list,
        csv,
        test=False,
    )

    # plot BTS vs bootstrap values
    # sort both dataframes so we plot corresponding values correctly
    branch_scores_df = branch_scores_df.sort_values(
        by=list(branch_scores_df.columns)
    ).reset_index(drop=True)
    full_tree_bootstrap_df = full_tree_bootstrap_df.sort_values(
        by=list(full_tree_bootstrap_df.columns)
    ).reset_index(drop=True)
    merged_df = pd.concat([branch_scores_df, full_tree_bootstrap_df], axis=1)

    # Compute the 2D histogram
    hist, xedges, yedges = np.histogram2d(
        merged_df["bootstrap_support"], merged_df["bts"], bins=50
    )

    # Use the histogram values to assign each data point a density value
    xidx = np.clip(
        np.digitize(merged_df["bootstrap_support"], xedges), 0, hist.shape[0] - 1
    )
    yidx = np.clip(np.digitize(merged_df["bts"], yedges), 0, hist.shape[1] - 1)
    merged_df["density"] = hist[xidx, yidx]

    # Create the scatter plot with colors based on density and a logarithmic scale
    plt.scatter(
        merged_df["bootstrap_support"],
        merged_df["bts"],
        c=merged_df["density"],
        cmap="RdBu_r",
    )

    cb = plt.colorbar(label="density")

    # Set other plot properties
    plt.title("BTS vs Bootstrap Support")
    plt.xlabel("bootstrap support in full tree")
    plt.ylabel("branch taxon score (bts)")
    plt.tight_layout()
    plt.savefig(bts_vs_bootstrap_path)
    plt.clf()


def df_column_swarmplot(csv, col_name, plot_filepath):
    """
    For each taxon, plot the value in column col_name vs TII values
    """
    df = pd.read_csv(csv)
    # Sort the DataFrame based on 'tii'
    df_sorted = df.sort_values(by="tii")

    plt.figure(figsize=(10, 6))
    sns.stripplot(data=df_sorted, x="seq_id", y=col_name)

    # Set labels and title
    plt.xlabel("taxa (sorted by TII)")
    plt.ylabel(col_name)
    plt.title(col_name + " vs. taxa sorted by TII")

    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(plot_filepath)
    plt.clf()


csv = snakemake.input.csv
full_tree_file = snakemake.input.full_tree
reattached_tree_files = snakemake.input.reattached_trees
reduced_tree_files = snakemake.input.reduced_trees
mldist_file = snakemake.input.mldist
reduced_mldist_files = snakemake.input.reduced_mldist

plots_folder = snakemake.params.plots_folder

if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

print("Start reading, aggregating, and filtering data.")
taxon_df = pd.read_csv(csv, index_col=0)
taxon_df.index.name = "taxon_name"

taxon_tii_list = [
    (seq_id.split(" ")[0], seq_id.split(" ")[1])
    for seq_id in taxon_df["seq_id"].unique()
]
sorted_taxon_tii_list = sorted(taxon_tii_list, key=lambda x: x[1])
print("Done reading data.")


print("Start plotting reattachment distances.")
plot_filepath = os.path.join(plots_folder, "dist_of_likely_reattachments.pdf")
plot_reattachment_distances(
    sorted_taxon_tii_list, reduced_tree_files, reattached_tree_files, plot_filepath
)
print("Done plotting reattachment distances.")


print("Start plotting reattachment distance to low support node.")
plot_filepath = os.path.join(
    plots_folder, "reattachment_distance_to_low_support_node.pdf"
)
reattachment_distance_to_low_support_node(
    sorted_taxon_tii_list, reattached_tree_files, bootstrap_threshold=1
)
print("Done plotting reattachment distance to low support node.")


print("Start plotting NJ TII.")
plot_filepath = os.path.join(plots_folder, "NJ_TII.pdf")
ratio_plot_filepath = os.path.join(plots_folder, "NJ_tree_likeness.pdf")
nj_tii(
    mldist_file,
    sorted_taxon_tii_list,
    reduced_mldist_files,
    plot_filepath,
    ratio_plot_filepath,
)
print("Done plotting NJ TII.")


print("Start plotting order of distances to seq_id in tree vs MSA.")
plot_filepath = os.path.join(plots_folder, "order_of_distances_to_seq_id.pdf")
barplot_filepath = os.path.join(plots_folder, "order_of_distances_to_seq_id_count.pdf")
histplot_filepath = os.path.join(
    plots_folder, "order_of_distances_to_seq_id_histplots.pdf"
)
order_of_distances_to_seq_id(
    sorted_taxon_tii_list,
    mldist_file,
    reattached_tree_files,
    plot_filepath,
    barplot_filepath,
)
print("Done plotting order of distances to seq_id in tree vs MSA.")


print("Start plotting sequence and tree distance differences.")
distance_diff_filepath = os.path.join(plots_folder, "seq_and_tree_dist_diff.pdf")
ratio_plot_filepath = os.path.join(plots_folder, "seq_and_tree_dist_ratio.pdf")
seq_and_tree_dist_diff(
    sorted_taxon_tii_list,
    mldist_file,
    reattached_tree_files,
    ratio_plot_filepath,
)
print("Done plotting sequence and tree distance differences.")

# # plot topological distance of reattachment locations vs TII, hue = log_likelihood
# # difference
# print("Start plotting topological reattachment distances.")
# reattachment_topological_distances_path = os.path.join(
#     plots_folder, "topological_dist_of_likely_reattachments.pdf"
# )
# dist_of_likely_reattachments(
#     sorted_taxon_tii_list,
#     best_two_taxon_edge_df,
#     reattachment_distance_topological_csv,
#     reattachment_topological_distances_path,
# )
# print("Done plotting topological reattachment distances.")


# swarmplot bootstrap support reduced tree for each taxon, sort by TII
print("Start plotting bootstrap and bts.")
bootstrap_plot_filepath = os.path.join(plots_folder, "bootstrap_vs_tii.pdf")
local_bootstrap_plot_filepath = os.path.join(plots_folder, "local_bootstrap_vs_tii.pdf")
bts_plot_filepath = os.path.join(plots_folder, "bts_scores.pdf")
bts_vs_bootstrap_path = os.path.join(plots_folder, "bts_vs_bootstrap.pdf")
bootstrap_and_bts_plot(
    reduced_tree_files,
    full_tree_file,
    sorted_taxon_tii_list,
    bts_vs_bootstrap_path,
    csv,
)
print("Done plotting bootstrap and bts.")


# Swarmplots of statistics we can get straight out of csv file
print("Start plotting likelihoods of reattached trees.")
ll_filepath = os.path.join(plots_folder, "likelihood_swarmplots.pdf")
df_column_swarmplot(csv, "likelihood", ll_filepath)
print("Done plotting likelihoods of reattached trees.")

print("Start plotting LWR of reattached trees.")
lwr_filepath = os.path.join(plots_folder, "likelihood_weight_ratio.pdf")
df_column_swarmplot(csv, "like_weight_ratio", lwr_filepath)
print("Done plotting LWR of reattached trees.")

print("Start plotting reattachment heights.")
taxon_height_plot_filepath = os.path.join(plots_folder, "taxon_height_vs_tii.pdf")
df_column_swarmplot(csv, "taxon_height", taxon_height_plot_filepath)
print("Done plotting reattachment heights.")

print("Start plotting reattachment branch length.")
reattachment_branch_length_plot_filepath = os.path.join(
    plots_folder, "reattachment_branch_length_vs_tii.pdf"
)
df_column_swarmplot(
    csv,
    "reattachment_branch_length",
    reattachment_branch_length_plot_filepath,
)
print("Done plotting reattachment branch length.")


print("Start plotting pendant branch length.")
pendant_branch_length_plot_filepath = os.path.join(
    plots_folder, "pendant_branch_length_vs_tii.pdf"
)
df_column_swarmplot(
    csv,
    "pendant_branch_length",
    pendant_branch_length_plot_filepath,
)
print("Done plotting reattachment branch length.")
