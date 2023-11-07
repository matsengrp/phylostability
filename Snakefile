from Bio import SeqIO


# input/output file names
input_alignment="input_alignment.fasta"
data_folder="data/"
plots_folder="plots/"
IQTREE_SUFFIXES=["iqtree", "log", "treefile", "ckp.gz"]


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
        expand(data_folder+"reduced_alignments/reattachment_data_per_taxon_epa.csv", seq_id = get_seq_ids(input_alignment))
        # "create_plots.done",
        # "create_other_plots.done"

# Define the rule to extract the best model for iqtree on the full MSA
rule model_test_iqtree:
    input:
        msa=input_alignment
    output:
        temp(touch(data_folder+"model-test-iqtree.done")),
        modeltest=data_folder+input_alignment+"_model.iqtree"
    shell:
        """
        if [[ -f "{data_folder}{input.msa}_model.iqtree" ]]; then
          echo "Ignoring iqtree ModelFinder run on {input.msa}, since it is already done."
        else
          iqtree -s {input.msa} --prefix {data_folder}{input.msa}_model -m MF
        fi
        """

# Define the rule to extract the model from the IQ-TREE run on the full msa
rule extract_model_for_full_iqtree_run:
    input:
        data_folder+"model-test-iqtree.done",
        msa=rules.model_test_iqtree.output.modeltest
    output:
        model=data_folder+"iqtree-model.txt"
    shell:
        """
        echo $(grep "Best-fit model" {input.msa} | cut -d ":" -f 2) > {output.model}
        """

# Define the rule to remove a sequence from the MSA and write the reduced MSA to a file
rule remove_sequence:
    input:
        msa=input_alignment
    output:
        reduced_msa=temp(data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta")
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
        temp(touch(data_folder+"run_iqtree_on_full_dataset.done")),
        tree=data_folder+input_alignment+".treefile",
        mldist=data_folder+input_alignment+".mldist"
    shell:
        """
        if [[ -f "{data_folder}{input.msa}.iqtree" ]]; then
          echo "Ignoring iqtree run on {input.msa}, since it is already done."
        else
          cp {input.msa} {data_folder}
          iqtree -s {input.msa} -m $(cat {input.full_model}) --prefix {data_folder}{input.msa} -bb 1000
        fi
        """


# Define the rule to run IQ-TREE on the reduced MSA
rule run_iqtree_restricted_alignments:
    input:
        reduced_msa=rules.remove_sequence.output.reduced_msa,
        full_model=rules.extract_model_for_full_iqtree_run.output.model
    output:
        done=temp(touch(data_folder+"reduced_alignments/{seq_id}/run_iqtree_restricted_alignments.done")),
        tree=data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile",
        mlfile=data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.iqtree"
    shell:
        """
        if [[ -f "{input.reduced_msa}.iqtree" ]]; then
          echo "Ignoring iqtree run on {input.reduced_msa}, since it is already done."
        else
          iqtree -s {input.reduced_msa} -m $(cat {input.full_model}) --prefix {input.reduced_msa} -bb 1000 -redo
        fi
        """


rule extract_single_fastas:
    input:
        msa=input_alignment
    output:
        taxon_msa=expand(data_folder+"reduced_alignments/{seq_id}/single_taxon.fasta", seq_id=get_seq_ids(input_alignment)),
        without_taxon_msa=expand(data_folder+"reduced_alignments/{seq_id}/without_taxon.fasta", seq_id=get_seq_ids(input_alignment))
    script:
        "scripts/extract_single_taxon_msa.py"


rule epa_reattachment:
    input:
        rules.run_iqtree_restricted_alignments.output.done,
        tree=data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile",
        model=data_folder+"iqtree-model.txt",
        taxon_msa=data_folder+"reduced_alignments/{seq_id}/single_taxon.fasta",
        without_taxon_msa=data_folder+"reduced_alignments/{seq_id}/without_taxon.fasta",
    output:
        epa_result=data_folder+"reduced_alignments/{seq_id}/epa_result.jplace",
    params:
        output_folder=data_folder+"reduced_alignments/{seq_id}"
    shell:
        """
        read -r model < {input.model}
        echo $model
        epa-ng --ref-msa {input.without_taxon_msa} --tree {input.tree} --query {input.taxon_msa} --model $model -w {params.output_folder} --redo
        """


rule extract_reattachment_statistics:
    input:
        epa_results=expand(data_folder+"reduced_alignments/{seq_id}/epa_result.jplace", seq_id = get_seq_ids(input_alignment)),
        full_tree=data_folder+input_alignment+".treefile"
    output:
        reattached_trees=expand(data_folder+"reduced_alignments/{seq_id}/reattached_tree.nwk", seq_id = get_seq_ids(input_alignment)),
        output_csv=data_folder+"reduced_alignments/reattachment_data_per_taxon_epa.csv"
    params:
        seq_ids=get_seq_ids(input_alignment),
    script:
        "scripts/extract_reattachment_statistics.py"


rule write_spr_tii_trees:
    input:
        full_tree=data_folder+input_alignment+".treefile",
    output:
        tree_pair_files=expand(data_folder+"reduced_alignments/{seq_id}/tii_trees.txt", seq_id = get_seq_ids(input_alignment))
    params:
        data_folder=data_folder
    script:
        "scripts/write_spr_tii_trees.py"


rule compute_spr_tii:
    input:
        tree_pair_files=data_folder+"reduced_alignments/{seq_id}/tii_trees.txt"
    output:
        spr_output=data_folder+"reduced_alignments/{seq_id}/spr_output.txt"
    shell:
        """
        ../uspr/uspr < {input.tree_pair_files} > {output.spr_output}
        """

rule add_spr_tii_to_df:
    input:
        csv=data_folder+"reduced_alignments/reattachment_data_per_taxon.csv",
        spr_files=expand(data_folder+"reduced_alignments/{seq_id}/spr_output.txt", seq_id = get_seq_ids(input_alignment))
    output:
        temp(touch("add_spr_tii_to_df.done")),
        csv=data_folder+"reduced_alignments/reattachment_data_per_taxon_spr_tii.csv",
    script:
        "scripts/add_spr_tii_to_df.py"


