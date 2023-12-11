import pandas as pd
import os
import math
from Bio.Nexus import Nexus

data_csv = snakemake.input.data_csv
data_folder = snakemake.params.data
selected_file = snakemake.params.selected_datasets_csv
N = snakemake.params.num_samples


# Function to check if Nexus file has alignment
def has_alignment(nexus_file):
    try:
        nexus = Nexus.Nexus(nexus_file)
        return bool(nexus.matrix)
    except Exception as e:
        print(f"Error reading Nexus file: {e}")
        return False


# Load existing selected datasets if the file exists
if os.path.isfile(selected_file):
    selected_datasets = pd.read_csv(selected_file)
else:
    selected_datasets = pd.DataFrame(columns=["taxa", "sequences", "file"])

# Load data
df = pd.read_csv(data_csv)

# Filter out already selected datasets
df = df[~df["file"].isin(selected_datasets["file"])]

# Define the number of bins
num_bins = math.floor(math.sqrt(N))

# Bin the data
df["taxa_bin"] = pd.qcut(df["taxa"], num_bins, duplicates="drop")
df["seq_bin"] = pd.qcut(df["sequences"], num_bins, duplicates="drop")

new_samples = df.groupby(["taxa_bin", "seq_bin"]).sample(n=1, random_state=1)

# Sample additional datasets
num_selected = len(new_samples)
if num_selected < N:
    additional_samples = df.drop(new_samples.index).sample(
        n=N - num_selected, random_state=1
    )
    # Verify Nexus files have alignments
    additional_samples = additional_samples[
        additional_samples["file"].apply(
            lambda x: has_alignment(os.path.join(data_folder, x))
        )
    ]
    new_samples = pd.concat([new_samples, additional_samples])

selected_datasets = pd.concat([selected_datasets, new_samples])

# Save the updated selected datasets
selected_datasets.to_csv(selected_file, index=False)

# Create the selected_data directory if it doesn't exist
selected_data_dir = "selected_data"
os.makedirs(os.path.join(data_folder, selected_data_dir), exist_ok=True)

# Iterate through the DataFrame and create symlinks
for index, row in selected_datasets.iterrows():
    original_file_path = os.path.join(data_folder, row["file"])
    filename = "_".join(row["file"].split("/"))
    this_row_dir = os.path.join(data_folder, selected_data_dir, filename[:-4])
    os.makedirs(this_row_dir, exist_ok=True)
    symlink_path = os.path.join(this_row_dir, os.path.basename(row["file"]))

    # Create a symbolic link if the original file exists
    if os.path.exists(original_file_path):
        if not os.path.islink(symlink_path):
            os.symlink(original_file_path, symlink_path)
    else:
        print(f"File not found: {original_file_path}")

print("Symbolic links created in selected_data directory.")
