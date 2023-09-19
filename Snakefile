import os
from Bio import SeqIO


# Retrieve all sequence IDs from the input multiple sequence alignment
def get_seq_ids(input_file):
    return [record.id for record in SeqIO.parse(input_file, "fasta")]


# Define the workflow
rule all:
    input:
        expand("reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile", seq_id=get_seq_ids("input_alignment.fasta"))


# Define the rule to remove a sequence from the MSA and write the reduced MSA to a file
rule remove_sequence:
    input:
        msa="input_alignment.fasta"
    output:
        reduced_msa=temp("reduced_alignments/{seq_id}/reduced_alignment.fasta")
    params:
        seq_id=lambda wildcards: wildcards.seq_id
    run:
        # Read the MSA
        with open(input.msa, "r") as f_in:
            sequences = list(SeqIO.parse(f_in, "fasta"))
        
        # Remove the sequence with the specified ID
        reduced_sequences = [seq for seq in sequences if seq.id != params.seq_id]
        
        # Create a directory for the reduced MSA if it doesn't exist
        os.makedirs(os.path.dirname(output.reduced_msa), exist_ok=True)
        
        # Write the reduced MSA to a file
        with open(output.reduced_msa, "w") as f_out:
            SeqIO.write(reduced_sequences, f_out, "fasta")


# Define the rule to run IQ-TREE on the reduced MSA
rule run_iqtree:
    input:
        reduced_msa=rules.remove_sequence.output.reduced_msa
    output:
        tree="reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile"
    shell:
        "iqtree -s {input.reduced_msa}"
