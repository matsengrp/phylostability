import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ete3 import Tree
import glob
import os
from Bio import SeqIO


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

def get_seq_ids(input_file, filetype):
    return [record.id for record in SeqIO.parse(input_file, filetype)]

# au_files = snakemake.input.au_files

# both_trees_file = snakemake.input.both_trees
au_test_results = snakemake.output.au_test_results
subdirs = snakemake.params.subdirs

au_files = []
for subdir in subdirs:
    fasta_files = glob.glob(os.path.join(subdir, "*.fasta"))
    fasta_files = [os.readlink(file) if os.path.islink(file) else file for file in fasta_files]
    seq_ids = get_seq_ids(fasta_files[0], "fasta")
    for seq_id in seq_ids:
        au_files.append(subdir+"/reduced_alignments/"+seq_id+"/pruned_and_inferred_tree.nwk")
        au_files.append(subdir+"/reduced_alignments/"+seq_id+"/au-test.iqtree")

print(au_files)

df_list = []

datasets = set([file.split("/")[2] for file in au_files])


for dataset in datasets:
    dataset_files = [f for f in au_files if dataset+"/" in f]
    for seq_id in [f.split("/")[4] for f in dataset_files]:
        this_seq_id_files = [f for f in dataset_files if seq_id in f]
        both_trees_file = [f for f in this_seq_id_files if "nwk" in f][0]
        iqtree_file = [f for f in this_seq_id_files if "iqtree" in f][0]
        df = extract_table_from_file(iqtree_file)
        df["dataset"] = dataset
        df["seq_id"] = seq_id
    
        with open(both_trees_file, "r") as f:
            trees = f.readlines()
        pruned_tree = Tree(trees[0].strip())
        inferred_tree = Tree(trees[1].strip())
        tii = pruned_tree.robinson_foulds(inferred_tree, unrooted_trees = True)[0]
        normalised_tii = tii/pruned_tree.robinson_foulds(inferred_tree, unrooted_trees = True)[1]
        df["tii"] = tii
        df["normalised_tii"] = normalised_tii
        df["ID"] = df["dataset"] + " " + df["seq_id"] + " " + df["tii"].astype(str)
        df_list.append(df)


big_df = pd.concat(df_list, ignore_index=True)
big_df.to_csv(au_test_results)
