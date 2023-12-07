import pandas as pd
import os
import sys
import math


data_folder = snakemake.params.data
N = snakemake.params.num_samples
data_csv = snakemake.input.data_csv

# Load data if not already in memory
df = pd.read_csv(data_csv)

# Define the number of bins
num_bins = math.floor(math.sqrt(N))

# Bin the data
df["taxa_bin"] = pd.qcut(df["taxa"], num_bins, duplicates="drop")
df["seq_bin"] = pd.qcut(df["sequences"], num_bins, duplicates="drop")

# Sample datasets
selected_datasets = df.groupby(["taxa_bin", "seq_bin"], observed=False).sample(
    n=1, random_state=1
)

# If the total selected datasets are less than 100, randomly select the remaining
num_selected = len(selected_datasets)
if num_selected < N:
    additional_samples = df.drop(selected_datasets.index).sample(
        n=N - num_selected, random_state=1
    )
    selected_datasets = pd.concat([selected_datasets, additional_samples])

# Save or process the selected datasets
selected_datasets.to_csv(data_folder + "/selected_datasets.csv", index=False)

# Create the selected_data directory if it doesn't exist
selected_data_dir = "selected_data"
os.makedirs(selected_data_dir, exist_ok=True)

# Iterate through the DataFrame and create symlinks
for index, row in selected_datasets.iterrows():
    original_file_path = os.path.join(data_folder + "/", row["file"])
    filename = "_".join(row["file"].split("/"))
    this_row_dir = filename[:-4]
    this_row_dir = data_folder + "/" + selected_data_dir + "/" + this_row_dir
    os.makedirs(this_row_dir, exist_ok=True)
    symlink_path = os.path.join(this_row_dir, os.path.basename(row["file"]))

    # Create a symbolic link if the original file exists
    if os.path.exists(original_file_path):
        os.symlink(original_file_path, symlink_path)
    else:
        print(f"File not found: {original_file_path}")

print("Symbolic links created in selected_data directory.")
