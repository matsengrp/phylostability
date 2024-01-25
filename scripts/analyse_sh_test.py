import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ete3 import Tree


iqtree_files = snakemake.input.iqtree_file
tree_order_files = snakemake.input.reattached_trees_order
summary_statistics = snakemake.input.summary_statistics
reattached_trees = snakemake.input.reattached_trees
classifier_statistics = snakemake.input.classifier_statistics
regression_statistics = snakemake.input.regression_statistics
plot_filepath = snakemake.output.plot
au_test_results = snakemake.output.au_test_results
au_test_regression_input = snakemake.output.au_test_regression_input
subdirs = snakemake.params.subdirs


def extract_table_from_file(filename):
    """
    Read table with AU-test results from iqtree run from .iqtree file.
    Returns table as pandas DataFrame.
    """
    # Flag to indicate whether we are currently reading the table
    reading_table = False
    table_data = []

    with open(filename, "r") as file:
        for line in file:
            # Check for the start of the table
            if line.startswith("Tree      logL"):
                reading_table = True
                continue  # Skip the header line

            # Check for the end of the table
            if line.startswith("deltaL"):
                break

            if line.startswith("------"):
                continue

            # Read table data
            if reading_table and line.strip():
                this_line = [l for l in line.split() if l not in ["+", "-"]]
                table_data.append(this_line)
    # Convert the table data to a pandas DataFrame
    columns = [
        "Tree",
        "logL",
        "deltaL",
        "bp-RELL",
        "p-KH",
        "p-SH",
        "p-WKH",
        "p-WSH",
        "c-ELW",
        "p-AU",
    ]
    df = pd.DataFrame(table_data, columns=columns)

    # Convert numerical columns from string to appropriate types
    for col in [
        "logL",
        "deltaL",
        "bp-RELL",
        "p-KH",
        "p-SH",
        "p-WKH",
        "p-WSH",
        "c-ELW",
        "p-AU",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


df_list = []
df_unfiltered_list = []
for subdir in subdirs:
    # get taxon name, normalised and unnormalised tii of subdir
    ss_df = pd.read_csv([f for f in summary_statistics if subdir + "/" in f][0])
    ss_df = ss_df[["seq_id", "normalised_tii", "tii"]]
    # make sure ss_df rows match those in other dfs (need to add one row for full tree)
    new_row = pd.DataFrame({"seq_id": ["full"], "normalised_tii": [0], "tii": [0]})
    ss_df = pd.concat([ss_df, new_row], ignore_index=True)

    # seq_id in ss_df is actually seq_id + " " + tii. We only want seq_id
    def extract_seq_id(s):
        return s.split(" ")[0]

    ss_df["seq_id"] = ss_df["seq_id"].apply(extract_seq_id)

    # extract iqtree output and corresponding seq_id order
    iqtree_file = [f for f in iqtree_files if subdir + "/" in f][0]
    order_file = [f for f in tree_order_files if subdir + "/" in f][0]
    with open(order_file, "r") as file:
        order = [line.strip() for line in file]

    # create df with AU-test results and additional information from ss_df
    df = extract_table_from_file(iqtree_file)
    df["dataset"] = iqtree_file.split("/")[-2]
    df["seq_id"] = order
    df = df.merge(ss_df, on="seq_id", how="left")
    df["ID"] = df["dataset"] + " " + df["seq_id"] + " " + df["tii"].astype(str)
    df_unfiltered_list.append(df)
    filtered_df = df[df["p-AU"] < 0.05]

    # iqtree deletes duplicate trees. If the tree with p-value < 0.05 is a duplicate,
    # we also want to add the remaining trees to the plot.
    reattached_tree_file = [f for f in reattached_trees if subdir + "/" in f][0]
    topology_id_map = {}
    id_to_topology = {}
    # get for each tree topology in reattached_tree_file all seq_ids for which reattached
    # trees have that topology (get maps in both ways)
    with open(order_file, "r") as ids_file, open(
        reattached_tree_file, "r"
    ) as trees_file:
        for id_line, tree_line in zip(ids_file, trees_file):
            tree_id = id_line.strip()
            newick_tree = Tree(tree_line.strip())
            tree_string = newick_tree.write(
                format=9
            )  # Convert the tree to a string representation (topology -- no branch lengths)
            if tree_string not in topology_id_map:
                topology_id_map[tree_string] = []
            topology_id_map[tree_string].append(tree_id)
            id_to_topology[id] = tree_string
    # collect topologies that are present in filtered_df, i.e. have AU-test p-value < 0.05
    topologies_of_interest = set()
    for tree_id in filtered_df["ID"]:
        for topology, ids in topology_id_map.items():
            if tree_id in ids:
                topologies_of_interest.add(topology)
                break
    # collect ids that match the topologies in filtered_df
    matching_ids = set(
        [topology_id_map[topology] for topology in topologies_of_interest]
    )
    # Add taxa with same topology to dataframe
    # Creating a copy to avoid modifying the original DataFrame while iterating
    new_rows = filtered_df.copy()

    for id in matching_ids:
        # Get the topology for this ID
        topology = id_to_topology[id]

        # Find a row in filtered_df with the same topology
        for original_id in topology_id_map[topology]:
            if original_id in filtered_df["ID"].values:
                tii = original_id.split(" ")[-1]
                # Copy the row and replace the ID
                row_copy = filtered_df[filtered_df["ID"] == original_id].copy()
                row_copy["ID"] = iqtree_file.split("/")[-2] + " " + id + " " + str(tii)
                new_rows = new_rows.append(row_copy, ignore_index=True)
                break

    # Now new_rows contains the original rows plus the new rows
    filtered_df = new_rows
    df_list.append(filtered_df)

# mean normalised tiis for regression and classifier data
classifier_df = pd.read_csv(classifier_statistics)
classifier_mean_tii = classifier_df["normalised_tii"].mean()
regression_df = pd.read_csv(regression_statistics)
regression_mean_tii = regression_df["normalised_tii"].mean()

big_df = pd.concat(df_list, ignore_index=True)
big_df.to_csv(au_test_results)
plt.figure(figsize=(10, 6))
ax = sns.histplot(big_df["normalised_tii"])
ax.axvline(classifier_mean_tii, color="black", label="Mean TII Classifier data")
ax.axvline(regression_mean_tii, color="red", label="Mean TII all data")
ax.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(plot_filepath)

all_au_df = pd.concat(df_unfiltered_list, ignore_index=True)
all_au_df.to_csv(au_test_regression_input)

# # Group by 'dataset' and calculate the proportion
# grouped_df = big_df.groupby("dataset", group_keys=True)
# proportion_df = grouped_df.apply(
#     lambda x: (x["p-AU"] < 0.05).sum() / len(x)
# ).reset_index(name="proportion")

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.bar(proportion_df["dataset"], proportion_df["proportion"])
# plt.xlabel("Dataset")
# plt.ylabel("Proportion of p-AU < 0.05")
# plt.xticks(rotation=45, ha="right")  # Rotate the x-axis labels for better readability
# plt.title("Proportion of p-AU < 0.05 for each file")
# plt.tight_layout()  # Adjust layout for better fit
# plt.savefig(plot_filepath)
# plt.clf()
