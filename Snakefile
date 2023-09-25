import os
from Bio import SeqIO

output_folder="data/"

# input/output file names
input_alignment="input_alignment.fasta"
output_folder="data/"

# Retrieve all sequence IDs from the input multiple sequence alignment
def get_seq_ids(input_file):
    return [record.id for record in SeqIO.parse(input_file, "fasta")]


# Define the workflow
rule all:
    input:
        expand(output_folder+"reduced_alignments/{seq_id}/restricted_tree.treefile", seq_id=get_seq_ids("input_alignment.fasta")),
        expand(output_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile", seq_id=get_seq_ids(input_alignment)),
        output_folder+input_alignment+".treefile"


# Define the rule to extract the best model for iqtree on the full MSA
rule model_test_iqtree:
    input:
        msa=input_alignment
    output:
        touch(output_folder+"model-test-iqtree.done"),
        modeltest=output_folder+"input_alignment.fasta_model.iqtree"
    shell:
        """
        iqtree -s {input.msa} --prefix {output_folder}{input.msa}_model -m MF
        """

# Define the rule to extract the model from the IQ-TREE run on the full msa
rule extract_model_for_full_iqtree_run:
    input:
        output_folder+"model-test-iqtree.done",
        msa=rules.model_test_iqtree.output.modeltest
    output:
        model=output_folder+"iqtree-model.txt"
    shell:
        """
        echo $(grep "Best-fit model" {input.msa} | cut -d ":" -f 2) > {output.model}
        """

# Define the rule to remove a sequence from the MSA and write the reduced MSA to a file
rule remove_sequence:
    input:
        msa=input_alignment
    output:
        reduced_msa=temp(output_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta")
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


rule get_restricted_trees:
    input:
        output_folder+"run_iqtree_on_full_dataset.done",
        full_tree=output_folder+"input_alignment.fasta.treefile"
    output:
        restricted_trees=expand(output_folder+"reduced_alignments/{seq_id}/restricted_tree.treefile", seq_id=get_seq_ids("input_alignment.fasta"))
    script:
        "scripts/create_restricted_trees.py"


# Define the rule to run IQ-TREE on the reduced MSA
rule run_iqtree_restricted_alignments:
    input:
        reduced_msa=rules.remove_sequence.output.reduced_msa,
        full_model=rules.extract_model_for_full_iqtree_run.output.model
    output:
        tree=output_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile"
    shell:
        """
        iqtree -s {input.reduced_msa} -m $(cat {input.full_model}) --prefix {input.reduced_msa} -bb 1000
        """


# Define the rule to run IQ-TREE on the full MSA and get model parameters
rule run_iqtree_on_full_dataset:
    input:
        msa=input_alignment,
        full_model=rules.extract_model_for_full_iqtree_run.output.model
    output:
        tree=output_folder+input_alignment+".treefile"
    shell:
        """
        cp {input.msa} {output_folder}
        iqtree -s {input.msa} -m $(cat {input.full_model}) --prefix {output_folder}{input.msa} -bb 1000
        """
