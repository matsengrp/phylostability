from Bio import SeqIO
import os

# Read the MSA
input_msa = snakemake.input.msa
output_msa = snakemake.output.reduced_msa
seq_id = snakemake.params.seq_id

with open(input_msa, "r") as f_in:
    sequences = list(SeqIO.parse(f_in, "fasta"))

# Remove the sequence with the specified ID
reduced_sequences = [seq for seq in sequences if seq.id != seq_id]

# Create a directory for the reduced MSA if it doesn't exist
os.makedirs(os.path.dirname(output_msa), exist_ok=True)

# Write the reduced MSA to a file
with open(output_msa, "w") as f_out:
    SeqIO.write(reduced_sequences, f_out, "fasta")
