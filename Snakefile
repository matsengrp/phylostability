import os
from Bio import SeqIO


# Retrieve all sequence IDs from the input multiple sequence alignment
def get_seq_ids(input_file):
    return [record.id for record in SeqIO.parse(input_file, "fasta")]


# Define the workflow
rule all:
    input:
        expand("data/reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile", seq_id=get_seq_ids("input_alignment.fasta"))


# Define the rule to extract the best model for iqtree on the full MSA
rule model_test_iqtree:
    input:
        msa="input_alignment.fasta"
    output:
        touch("data/model-test-iqtree.done"),
        modeltest="data/input_alignment.fasta_model.iqtree"
    shell:
        """
        iqtree -s {input.msa} --prefix data/{input.msa}_model -m MF
        """

# Define the rule to extract the model from the IQ-TREE run on the full msa
rule extract_model_for_full_iqtree_run:
    input:
        "data/model-test-iqtree.done",
        msa=rules.model_test_iqtree.output.modeltest
    output:
        model="data/iqtree-model.txt"
    shell:
        """
        echo $(grep "Best-fit model" {input.msa} | cut -d ":" -f 2) > {output.model}
        """

# Define the rule to run IQ-TREE on the full MSA and get model parameters
rule run_iqtree_on_full_dataset:
    input:
        msa="input_alignment.fasta"
    output:
        filename="data/input_alignment.fasta.iqtree"
    shell:
        """
        iqtree -s {input.msa} --prefix data/{input.msa}
        """

# Define the rule to remove a sequence from the MSA and write the reduced MSA to a file
rule remove_sequence:
    input:
        msa="input_alignment.fasta"
    output:
        reduced_msa=temp("data/reduced_alignments/{seq_id}/reduced_alignment.fasta")
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
        reduced_msa=rules.remove_sequence.output.reduced_msa,
        full_model=rules.extract_model_for_full_iqtree_run.output.model
    output:
        tree="data/reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile"
    shell:
        """
        iqtree -s {input.reduced_msa} -m $(cat {input.full_model}) --prefix {input.reduced_msa}
        """
