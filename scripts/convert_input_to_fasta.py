from Bio import SeqIO
from Bio.Nexus import Nexus
import os
import glob
import shutil

# Path to the data directory
data_dir = snakemake.params.data_folder


def is_alignment(input_file):
    with open(input_file, "r") as handle:
        # Attempt to parse the file as a FASTA
        alignment = []
        if "fasta" in input_file:
            alignment = list(SeqIO.parse(handle, "fasta"))
        elif "nex" in input_file:
            alignment = list(SeqIO.parse(handle, "nexus"))
        if len(alignment) > 0:
            return True  # Successfully parsed as FASTA with sequences
    return False  # No sequences found


for subdir in [
    f.path
    for f in os.scandir(data_dir)
    if f.is_dir() and "plot" not in f.path and "benchmarking" not in f.path
]:
    msa_file = subdir + "/full_alignment.fasta"
    if os.path.exists(msa_file):
        if not is_alignment(msa_file):
            print(f"full_alignment.fasta doesn't contain msa. Delete {subdir}")
            shutil.rmtree(subdir)
            continue
        continue
    # Search for Nexus and FASTA files
    nexus_files = [
        f for f in glob.glob(os.path.join(subdir, "*.nex")) if ".splits." not in f
    ]
    fasta_files = glob.glob(os.path.join(subdir, "*.fasta"))
    # resolve symlinks
    nexus_files = [
        os.readlink(file) if os.path.islink(file) else file for file in nexus_files
    ]
    fasta_files = [
        os.readlink(file) if os.path.islink(file) else file for file in fasta_files
    ]
    nexus_file = None
    if len(fasta_files) == 0 and len(nexus_files) == 0:
        print(f"No alignment files in {subdir}. Delete {subdir}.")
        shutil.rmtree(subdir)
        continue
    if len(fasta_files) > 0:
        fasta_file = fasta_files[0]
        if not is_alignment(fasta_file):
            print(f"{fasta_file} doesn't contain alignment. Delete {subdir}.")
            shutil.rmtree(subdir)
            continue
        os.rename(fasta_file, msa_file)
        print(f"Renamed {fasta_file} to {msa_file}")
        continue

    elif len(nexus_files) == 1:
        # If there is only the .n.nex file, we convert that one
        # Otherwise, we go to 'else' and convert the other nexus file
        nexus_file = nexus_files[0]
        if not is_alignment(nexus_file):
            print(f"{nexus_file} doesn't contain alignment. Delete {subdir}.")
            shutil.rmtree(subdir)
            continue
        SeqIO.convert(nexus_file, "nexus", msa_file, "fasta")
    else:
        nexus_file = [f for f in nexus_files if ".n." not in f and ".splits." not in f][
            0
        ]
        if not is_alignment(nexus_file):
            shutil.rmtree(subdir)
            continue
        SeqIO.convert(nexus_file, "nexus", msa_file, "fasta")

    # If sequences are labelled by integers, we update those labels to s_1, s_2, ...
    # to make sure that we don't get problems with reading trees with bootstrap support
    # or internal node names later.

    def is_integer(label):
        try:
            int(label)
            return True
        except ValueError:
            return False

    # Read the FASTA file
    if nexus_file != None:
        records = list(SeqIO.parse(nexus_file, "nexus"))
    else:
        # This is if we already had fasta file beforehand
        record = list(SeqIO.parse(fasta_file, "fasta"))

    # Store unique sequences
    seen_sequences = set()
    unique_records = []

    # Modify the labels if they are integers and remove duplicates
    for i, record in enumerate(records):
        if is_integer(record.id):
            record.id = f"s_{i + 1}"
            record.description = f"s_{i + 1}"

        # Add unique sequences to new list
        if str(record.seq) not in seen_sequences:
            seen_sequences.add(str(record.seq))
            unique_records.append(record)

    # Write the modified records to a new FASTA file
    SeqIO.write(unique_records, msa_file, "fasta")
