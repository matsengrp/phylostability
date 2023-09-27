import os
from Bio import SeqIO
from ete3 import Tree


# input/output file names
input_alignment="input_alignment.fasta"
output_folder="data/"
IQTREE_SUFFIXES=["iqtree", "log", "treefile", "ckp.gz"]


# dictionary to hold the outputs of rules reattach_removed_sequence
sequence_reattachment_data = {}
data_for_each_taxon = {}

 
# Retrieve all sequence IDs from the input multiple sequence alignment
def get_seq_ids(input_file):
    return [record.id for record in SeqIO.parse(input_file, "fasta")]


def get_attachment_edge_indices(input_file):
    num_sequences = 0
    for record in SeqIO.parse(input_file, "fasta"):
        num_sequences += 1
    return range(1, 2*(num_sequences-1)-2)


# Define the workflow
rule all:
    input:
        expand(output_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta_add_at_edge_{edge}.run_iqtree.done", seq_id=get_seq_ids(input_alignment), edge=get_attachment_edge_indices(input_alignment)),
        expand(output_folder+"reduced_alignments/{seq_id}/restricted_tree.treefile", seq_id=get_seq_ids(input_alignment)),
        expand(output_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile", seq_id=get_seq_ids(input_alignment)),
        expand(output_folder+"reduced_alignments/{seq_id}/extract_reattachment_data_per_taxon_and_edge.csv", seq_id = get_seq_ids(input_alignment)),
        output_folder+"reduced_alignments/reattachment_data_per_taxon.csv",
        output_folder+input_alignment+".treefile"


# Define the rule to extract the best model for iqtree on the full MSA
rule model_test_iqtree:
    input:
        msa=input_alignment
    output:
        temp(touch(output_folder+"model-test-iqtree.done")),
        modeltest=output_folder+"input_alignment.fasta_model.iqtree"
    shell:
        """
        if [[ -f "{output_folder}{input.msa}_model.iqtree" ]]; then
          echo "Ignoring iqtree ModelFinder run on {input.msa}, since it is already done."
        else
          iqtree -s {input.msa} --prefix {output_folder}{input.msa}_model -m MF
        fi
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
    script:
        "scripts/remove_sequence.py"


# Define the rule to run IQ-TREE on the full MSA and get model parameters
rule run_iqtree_on_full_dataset:
    input:
        msa=input_alignment,
        full_model=rules.extract_model_for_full_iqtree_run.output.model
    output:
        temp(touch(output_folder+"run_iqtree_on_full_dataset.done")),
        tree=output_folder+input_alignment+".treefile"
    shell:
        """
        if [[ -f "{output_folder}{input.msa}.iqtree" ]]; then
          echo "Ignoring iqtree run on {input.msa}, since it is already done."
        else
          cp {input.msa} {output_folder}
          iqtree -s {input.msa} -m $(cat {input.full_model}) --prefix {output_folder}{input.msa}
        fi
        """


rule get_restricted_trees:
    input:
        output_folder+"run_iqtree_on_full_dataset.done",
        full_tree=output_folder+"input_alignment.fasta.treefile"
    output:
        restricted_trees=expand(output_folder+"reduced_alignments/{seq_id}/restricted_tree.treefile", seq_id=get_seq_ids(input_alignment))
    script:
        "scripts/create_restricted_trees.py"


# Define the rule to run IQ-TREE on the reduced MSA
rule run_iqtree_restricted_alignments:
    input:
        reduced_msa=rules.remove_sequence.output.reduced_msa,
        full_model=rules.extract_model_for_full_iqtree_run.output.model
    output:
        done=temp(touch(output_folder+"reduced_alignments/{seq_id}/run_iqtree_restricted_alignments.done")),
        tree=output_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile"
    shell:
        """
        if [[ -f "{input.reduced_msa}.iqtree" ]]; then
          echo "Ignoring iqtree run on {input.reduced_msa}, since it is already done."
        else
          iqtree -s {input.reduced_msa} -m $(cat {input.full_model}) --prefix {input.reduced_msa}
        fi
        """


# Define the rule to attach the pruned taxon at each edge
rule reattach_removed_sequence:
    input:
        rules.run_iqtree_restricted_alignments.output.done,
        reduced_tree_nwk=rules.run_iqtree_restricted_alignments.output.tree
    output:
        topologies=expand(output_folder+"reduced_alignments/{{seq_id}}/reduced_alignment.fasta_add_at_edge_{edge}.nwk", edge=get_attachment_edge_indices(input_alignment))
    params:
        seq_id=lambda wildcards: wildcards.seq_id
    script:
        "scripts/reattach_removed_sequence.py"


rule run_iqtree_on_augmented_topologies:
   input:
       msa=input_alignment,
       topology_file=output_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta_add_at_edge_{edge}.nwk",
       full_model=rules.extract_model_for_full_iqtree_run.output.model
   output:
       alldone=temp(touch(output_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta_add_at_edge_{edge}.run_iqtree.done")),
       treefile=output_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta_add_at_edge_{edge}.nwk_branch_length.treefile",
       mlfile=temp(output_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta_add_at_edge_{edge}.nwk_branch_length.iqtree"),
       other=temp(expand(output_folder+"reduced_alignments/{{seq_id}}/reduced_alignment.fasta_add_at_edge_{{edge}}.nwk_branch_length.{suffix}",
        suffix=[suf for suf in IQTREE_SUFFIXES if suf not in ["iqtree", "treefile"]]))
   shell:
       """
       if test -f "{input.topology_file}_branch_length.iqtree"; then
         echo "Ignoring iqtree run on {input.topology_file}_branch_length, since it is already done."
       else
         iqtree -s {input.msa} -m $(cat {input.full_model}) -te {input.topology_file} --prefix {input.topology_file}_branch_length
       fi
        """


# this rule adds a specific key to the global dictionary
rule extract_reattachment_data_per_taxon_and_edge:
    input:
        ready_to_run=expand(output_folder+"reduced_alignments/{{seq_id}}/reduced_alignment.fasta_add_at_edge_{edge}.run_iqtree.done", edge=get_attachment_edge_indices(input_alignment)),
        treefiles=expand(output_folder+"reduced_alignments/{{seq_id}}/reduced_alignment.fasta_add_at_edge_{edge}.nwk_branch_length.treefile", edge=get_attachment_edge_indices(input_alignment)),
        mlfiles=expand(output_folder+"reduced_alignments/{{seq_id}}/reduced_alignment.fasta_add_at_edge_{edge}.nwk_branch_length.iqtree", edge=get_attachment_edge_indices(input_alignment)),
        full_tree_file=rules.run_iqtree_on_full_dataset.output.tree
    output:
        csv_name=output_folder+"reduced_alignments/{seq_id}/extract_reattachment_data_per_taxon_and_edge.csv"
    params:
        seq_id=lambda wildcards: wildcards.seq_id
    script:
        "scripts/extract_reattachment_data_per_taxon_and_edge.py"


rule aggregate_reattachment_data_per_taxon:
    input:
        treefiles=expand(output_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta_add_at_edge_{edge}.nwk_branch_length.treefile", edge=get_attachment_edge_indices(input_alignment), seq_id=get_seq_ids(input_alignment)),
        reduced_treefile=expand(output_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile", seq_id=get_seq_ids(input_alignment)),
        taxon_dictionary=expand(output_folder+"reduced_alignments/{seq_id}/extract_reattachment_data_per_taxon_and_edge.csv", seq_id=get_seq_ids(input_alignment))
    output:
        output_csv=output_folder+"reduced_alignments/reattachment_data_per_taxon.csv"
    params:
        seq_ids=get_seq_ids(input_alignment),
        edges=get_attachment_edge_indices(input_alignment),
    script:
        "scripts/aggregate_reattachment_data_per_taxon.py"
