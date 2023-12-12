from Bio import SeqIO
from Bio.Nexus import Nexus
import os
import glob

# Path to the data directory
data_dir = snakemake.input.data_folder
msa_file = snakemake.output.input_alignment
subdir = snakemake.params.subdir


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

if len(fasta_files) > 0:
    fasta_file = fasta_files[0]
    os.rename(fasta_file, msa_file)
    print(f"Renamed {fasta_file} to {msa_file}")

elif len(nexus_files) == 1:
    # If there is only the .n.nex file, we convert that one
    # Otherwise, we go to 'else' and convert the other nexus file
    nexus_file = nexus_files[0]
    SeqIO.convert(nexus_file, "nexus", msa_file, "fasta")
else:
    nexus_file = [f for f in nexus_files if ".n." not in f and ".splits." not in f][0]
    SeqIO.convert(nexus_file, "nexus", msa_file, "fasta")

# If sequences are labelles by integers, we update those labels to s_1, s_2, ...
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
