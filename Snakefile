from Bio import SeqIO


# input/output file names
input_alignment="input_alignment.fasta"
data_folder="data/"
plots_folder="plots/epa/"
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
        "create_plots.done",
        # "create_other_plots.done",


# Define the rule to extract the best model for iqtree on the full MSA
rule model_test_iqtree:
    input:
        msa=input_alignment
    output:
        temp(touch(data_folder+"model-test-iqtree.done")),
        modeltest=data_folder+input_alignment+"_model.iqtree"
    shell:
        """
        iqtree -s {input.msa} --prefix {data_folder}{input.msa}_model -m MF -redo
        """


# Define the rule to extract the model from the IQ-TREE run on the full msa
rule extract_model_for_full_iqtree_run:
    input:
        data_folder+"model-test-iqtree.done",
        iqtree=rules.model_test_iqtree.output.modeltest,
    output:
        model=data_folder+"iqtree-model.txt"
    script:
        "scripts/extract_model.py"


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
        cp {input.msa} {data_folder}
        iqtree -s {input.msa} -m $(cat {input.full_model}) --prefix {data_folder}{input.msa} -bb 1000 -redo
        """


# Define the rule to run IQ-TREE on the reduced MSA
rule run_iqtree_restricted_alignments:
    input:
        reduced_msa=rules.remove_sequence.output.reduced_msa,
        full_model=rules.extract_model_for_full_iqtree_run.output.model
    output:
        done=temp(touch(data_folder+"reduced_alignments/{seq_id}/run_iqtree_restricted_alignments.done")),
        tree=data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile",
        mlfile=data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.iqtree",
        mldist=data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.mldist"
    shell:
        """
        iqtree -s {input.reduced_msa} -m $(cat {input.full_model}) --prefix {input.reduced_msa} -bb 1000 -redo
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
        model=rules.extract_model_for_full_iqtree_run.output.model,
        taxon_msa=data_folder+"reduced_alignments/{seq_id}/single_taxon.fasta",
        without_taxon_msa=data_folder+"reduced_alignments/{seq_id}/without_taxon.fasta",
    output:
        epa_result=data_folder+"reduced_alignments/{seq_id}/epa_result.jplace",
    params:
        output_folder=data_folder+"reduced_alignments/{seq_id}"
    shell:
        """
        model=$(cat {input.model})
        epa-ng --ref-msa {input.without_taxon_msa} --tree {input.tree} --query {input.taxon_msa} --model $model -w {params.output_folder} --redo
        """


rule extract_reattachment_statistics:
    input:
        epa_results=expand(data_folder+"reduced_alignments/{seq_id}/epa_result.jplace", seq_id = get_seq_ids(input_alignment)),
        restricted_trees=expand(data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile",seq_id=get_seq_ids(input_alignment)),
        full_tree=data_folder+input_alignment+".treefile",
        full_mldist_file=data_folder+input_alignment+".mldist",
        restricted_mldist_files=expand(data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.mldist",seq_id=get_seq_ids(input_alignment)),
    output:
        reattached_trees=expand(data_folder+"reduced_alignments/{seq_id}/reattached_tree.nwk", seq_id = get_seq_ids(input_alignment)),
        plot_csv=data_folder+"reduced_alignments/reattachment_data_per_taxon_epa.csv",
        random_forest_csv=data_folder+"reduced_alignments/random_forest_input.csv",
    params:
        seq_ids=get_seq_ids(input_alignment),
    script:
        "scripts/extract_reattachment_statistics.py"


rule random_forest_regression:
    input:
        csv=data_folder+"reduced_alignments/random_forest_input.csv",
        epa_results=expand(data_folder+"reduced_alignments/{seq_id}/epa_result.jplace", seq_id = get_seq_ids(input_alignment))
    output:
        model_features_file=temp("model_feature_importances.csv"),
        output_file_name=temp("random_forest_regression.csv")
    params:
        column_to_predict = "tii",
        model_features_csv="model_feature_importances.csv",
        output_file_name="random_forest_regression.csv"
    script:
        "scripts/random_forest_regression.py"


rule random_forest_classifier:
    input:
        csv=data_folder+"reduced_alignments/random_forest_input.csv",
        epa_results=expand(data_folder+"reduced_alignments/{seq_id}/epa_result.jplace", seq_id = get_seq_ids(input_alignment))
    output:
        model_features_file=temp("discrete_model_feature_importances.csv"),
        output_file_name=temp("random_forest_classification.csv")
    params:
        column_to_predict = "tii",
        model_features_csv="discrete_model_feature_importances.csv",
        output_file_name="random_forest_classification.csv"
    script:
        "scripts/random_forest_classifier.py"


rule create_plots:
    input:
        csv=data_folder+"reduced_alignments/reattachment_data_per_taxon_epa.csv",
        random_forest_regression_csv=rules.random_forest_regression.output.output_file_name,
        model_features_csv=rules.random_forest_regression.output.model_features_file,
        random_forest_classifier_csv=rules.random_forest_classifier.output.output_file_name,
        discrete_model_features_csv=rules.random_forest_classifier.output.model_features_file,
        full_tree=data_folder+input_alignment+".treefile",
        reduced_trees=expand(data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile", seq_id=get_seq_ids(input_alignment)),
        reattached_trees=expand(data_folder+"reduced_alignments/{seq_id}/reattached_tree.nwk", seq_id=get_seq_ids(input_alignment)),
        mldist=data_folder+input_alignment+".mldist",
        reduced_mldist=expand(data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.mldist", seq_id=get_seq_ids(input_alignment)),
    output:
        temp(touch("create_plots.done")),
    params:
        plots_folder=plots_folder
    script:
        "scripts/create_plots.py"

        
rule create_other_plots:
    input:
        csv=data_folder+"reduced_alignments/reattachment_data_per_taxon_epa.csv",
        full_tree=data_folder+input_alignment+".treefile",
        reduced_trees=expand(data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile", seq_id=get_seq_ids(input_alignment)),
        reattached_trees=expand(data_folder+"reduced_alignments/{seq_id}/reattached_tree.nwk", seq_id=get_seq_ids(input_alignment)),
        mldist=data_folder+input_alignment+".mldist",
        reduced_mldist=expand(data_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.mldist", seq_id=get_seq_ids(input_alignment)),
    output:
        temp(touch("create_other_plots.done")),
    params:
        plots_folder=plots_folder
    script:
        "scripts/create_other_plots.py"
