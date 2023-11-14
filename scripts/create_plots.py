import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import ast

from utils import *


def get_bootstrap_and_bts_scores(
    reduced_tree_files,
    full_tree_file,
    sorted_taxon_tii_list,
    csv,
    test=False,
):
    """
    Returns DataFrame branch_scores_df containing bts values for all edges in tree in full_tree_file
    """
    csv = pd.read_csv(csv)

    full_tree = Tree(full_tree_file)
    num_leaves = len(full_tree)

    # extract bootstrap support from full tree
    bootstrap_dict = {
        ",".join(sorted(node.get_leaf_names())): node.support
        for node in full_tree.iter_descendants()
        if not node.is_leaf()
    }
    bootstrap_df = pd.DataFrame(bootstrap_dict, index=["bootstrap_support"]).transpose()

    # initialise branch score dict with sets of leaves representing edges of
    # full_tree
    branch_scores = {}
    all_taxa = full_tree.get_leaf_names()
    branch_scores = {
        ",".join(sorted(node.get_leaf_names())): 0
        for node in full_tree.iter_descendants()
        if not node.is_leaf()
    }

    for treefile in reduced_tree_files:
        tree = Tree(treefile)
        seq_id = treefile.split("/")[-2]
        tii = [p[1] for p in sorted_taxon_tii_list if p[0] == seq_id][0]

        # collect bootstrap support and bts for all nodes in tree
        for node in tree.traverse("postorder"):
            if not node.is_leaf() and not node.is_root():
                # add bootstrap support values for seq_id
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

    # sort both dataframes so we plot corresponding values correctly
    branch_scores_df = branch_scores_df.sort_values(
        by=list(branch_scores_df.columns)
    ).reset_index(drop=True)
    bootstrap_df = bootstrap_df.sort_values(by=list(bootstrap_df.columns)).reset_index(
        drop=True
    )
    merged_df = pd.concat([branch_scores_df, bootstrap_df], axis=1)

    if test == True:
        with open("test_data/bts_df.p", "wb") as f:
            pickle.dump(branch_scores_df, file=f)

    return merged_df


def bootstrap_and_bts_plot(
    reduced_tree_files,
    full_tree_file,
    sorted_taxon_tii_list,
    plot_filepath,
    csv,
):
    """
    Plot BTS vs bootstrap support values of full tree
    """
    # plot BTS vs bootstrap values
    merged_df = get_bootstrap_and_bts_scores(
        reduced_tree_files,
        full_tree_file,
        sorted_taxon_tii_list,
        csv,
        test=False,
    )

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
    plt.savefig(plot_filepath)
    plt.clf()


def df_column_swarmplot(csv, col_name, plot_filepath):
    """
    For each taxon, plot the value in column col_name vs TII values
    """
    df = pd.read_csv(csv)

    # Check if the column contains string representations of lists
    if isinstance(df.iloc[0][col_name], str):
        df[col_name] = df[col_name].apply(ast.literal_eval)
        df = df.explode(col_name)

    # Convert the column to numeric, to handle any inadvertent string or object types
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
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
    (seq_id.split(" ")[0], int(seq_id.split(" ")[1]))
    for seq_id in taxon_df["seq_id"].unique()
]
sorted_taxon_tii_list = sorted(taxon_tii_list, key=lambda x: x[1])
print("Done reading data.")


print("Start plotting reattachment distances.")
plot_filepath = os.path.join(plots_folder, "dist_of_likely_reattachments.pdf")
df_column_swarmplot(csv, "reattachment_distances", plot_filepath)
print("Done plotting reattachment distances.")


print("Start plotting reattachment distance to low support node.")
plot_filepath = os.path.join(
    plots_folder, "reattachment_distance_to_low_support_node.pdf"
)
df_column_swarmplot(csv, "dist_reattachment_low_bootstrap_node", plot_filepath)
print("Done plotting reattachment distance to low support node.")


print("Start plotting NJ TII.")
plot_filepath = os.path.join(plots_folder, "NJ_TII.pdf")
df_column_swarmplot(csv, "nj_tii", plot_filepath)
print("Done plotting NJ TII.")


print("Start plotting order of distances to seq_id in tree vs MSA.")
plot_filepath = os.path.join(plots_folder, "order_of_distances_to_seq_id.pdf")
df_column_swarmplot(csv, "order_diff", plot_filepath)
print("Done plotting order of distances to seq_id in tree vs MSA.")


print("Start plotting sequence and tree distance differences.")
plot_filepath = os.path.join(plots_folder, "seq_and_tree_dist_ratio.pdf")
df_column_swarmplot(csv, "seq_and_tree_dist_ratio", plot_filepath)
print("Done plotting sequence and tree distance differences.")


# swarmplot bootstrap support reduced tree for each taxon, sort by TII
print("Start plotting bootstrap and bts.")
plot_filepath = os.path.join(plots_folder, "bts_vs_bootstrap.pdf")
bootstrap_and_bts_plot(
    reduced_tree_files,
    full_tree_file,
    sorted_taxon_tii_list,
    plot_filepath,
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
