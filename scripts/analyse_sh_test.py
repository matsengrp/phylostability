import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


iqtree_files = snakemake.input.iqtree_file
tree_order_files = snakemake.input.reattached_trees_order
summary_statistics = snakemake.input.summary_statistics
plot_filepath = snakemake.output.plot
subdirs = snakemake.params.subdirs


def extract_table_from_file(filename):
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


# Get order of trees in iqtree table

df_list = []
for subdir in subdirs:
    ss_df = pd.read_csv([f for f in summary_statistics if subdir + "/" in f][0])
    ss_df = ss_df[["seq_id", "normalised_tii", "tii"]]
    # make sure ss_df rows match those in other dfs (need to add one row for full tree)
    new_row = pd.DataFrame({'seq_id': ["full"], 'normalised_tii': [0], 'tii': [0]})
    ss_df = pd.concat([ss_df, new_row], ignore_index=True)
    # seq_id in ss_df is actually seq_id + " " + tii. We need to fix that.
    def extract_seq_id(s):
        return s.split(" ")[0]
    ss_df['seq_id'] = ss_df['seq_id'].apply(extract_seq_id)

    # extract iqtree output and corresponding seq_id order
    iqtree_file = [f for f in iqtree_files if subdir + "/" in f][0]
    order_file = [f for f in tree_order_files if subdir + "/" in f][0]
    with open(order_file, "r") as file:
        order = [line.strip() for line in file]

    df = extract_table_from_file(iqtree_file)
    df["dataset"] = iqtree_file.split("/")[-2]
    df["seq_id"] = order
    df = df.merge(ss_df, on='seq_id', how='left')
    df["ID"] = df["dataset"] + " " + df["seq_id"] + " " +   df["tii"].astype(str)
    df_list.append(df)

big_df = pd.concat(df_list, ignore_index=True)

filtered_df = big_df[big_df["p-AU"] < 0.05]
plt.figure(figsize=(10,6))
sns.scatterplot(filtered_df, x = "ID", y = "normalised_tii")
plt.savefig(plot_filepath)

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