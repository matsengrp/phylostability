from Bio import SeqIO
import os
import glob

# input/output file names
input_alignment="full_alignment.fasta"
data_folder="harrington_data/selected_data/"
plots_folder="plots/"
IQTREE_SUFFIXES=["iqtree", "log", "treefile", "ckp.gz"]

def get_subdirs(data_folder):
    return [f.path for f in os.scandir(data_folder) if f.is_dir() and "plot" not in f.path and "benchmarking" not in f.path]

# Retrieve all sequence IDs from the input multiple sequence alignment
def get_seq_ids(input_file, filetype):
    return [record.id for record in SeqIO.parse(input_file, filetype)]

def dynamic_input(wildcards):
    subdir = wildcards.subdir
    nexus_files = glob.glob(os.path.join(subdir, "*.nex"))
    fasta_files = glob.glob(os.path.join(subdir, "*.fasta"))
    nexus_files = [os.readlink(file) if os.path.islink(file) else file for file in nexus_files]
    fasta_files = [os.readlink(file) if os.path.islink(file) else file for file in fasta_files]

    if len(fasta_files) > 0:
        seq_ids = get_seq_ids(fasta_files[0], "fasta")
    else:
        seq_ids = get_seq_ids(nexus_files[0], "nexus")

    epa_results = [subdir+"/reduced_alignments/"+seq_id+"/epa_result.jplace" for seq_id in seq_ids]
    restricted_trees = [subdir+"/reduced_alignments/"+seq_id+"/reduced_alignment.fasta.treefile" for seq_id in seq_ids]
    restricted_mldist_files = [subdir+"/reduced_alignments/"+seq_id+"/reduced_alignment.fasta.mldist" for seq_id in seq_ids]
    reattached_trees = [subdir+"/reduced_alignments/"+seq_id+"/reattached_tree.nwk" for seq_id in seq_ids]
    return epa_results + restricted_trees + restricted_mldist_files + reattached_trees


# Define the workflow
rule all:
    input:
        "random_forest_plots.done",
        # expand("{subdir}/create_plots.done", subdir=subdirs),
        # expand("{subdir}/create_other_plots.done", subdir=subdirs)


# convert input alignments from nexus to fasta, if necessary
rule convert_input_to_fasta:
    input:
        data_folder=data_folder,
    output:
        temp(touch("{subdir}/convert_input_to_fasta.done")),
        input_alignment="{subdir}/"+input_alignment,
    params:
        subdir=lambda wildcards: wildcards.subdir
    script:
        "scripts/convert_input_to_fasta.py"


# Define the rule to extract the best model for iqtree on the full MSA
rule model_test_iqtree:
    input:
        "{subdir}/convert_input_to_fasta.done",
        msa="{subdir}/"+input_alignment
    output:
        temp(touch("{subdir}/model-test-iqtree.done")),
        modeltest="{subdir}/"+input_alignment+"_model.iqtree"
    shell:
        """
        iqtree -s {input.msa} --prefix {input.msa}_model -m MF -redo
        """


# Define the rule to extract the model from the IQ-TREE run on the full msa
rule extract_model_for_full_iqtree_run:
    input:
        "{subdir}/model-test-iqtree.done",
        iqtree=rules.model_test_iqtree.output.modeltest,
    output:
        model="{subdir}/iqtree-model.txt"
    script:
        "scripts/extract_model.py"


# Define the rule to remove a sequence from the MSA and write the reduced MSA to a file
rule remove_sequence:
    input:
        "{subdir}/convert_input_to_fasta.done",
        msa="{subdir}/"+input_alignment
    output:
        reduced_msa=temp("{subdir}/reduced_alignments/{seq_id}/reduced_alignment.fasta"),
    params:
        seq_id=lambda wildcards: wildcards.seq_id
    script:
        "scripts/remove_sequence.py"


# Define the rule to run IQ-TREE on the full MSA and get model parameters
rule run_iqtree_on_full_dataset:
    input:
        "{subdir}/convert_input_to_fasta.done",
        msa="{subdir}/"+input_alignment,
        full_model="{subdir}/iqtree-model.txt"
    output:
        temp(touch("{subdir}/run_iqtree_on_full_dataset.done")),
        tree="{subdir}/"+input_alignment+".treefile",
        mldist="{subdir}/"+input_alignment+".mldist"
    shell:
        """
        iqtree -s {input.msa} -m $(cat {input.full_model}) --prefix {input.msa} -bb 1000 -redo
        """



# Define the rule to run IQ-TREE on the reduced MSA
rule run_iqtree_restricted_alignments:
    input:
        reduced_msa=rules.remove_sequence.output.reduced_msa,
        full_model=rules.extract_model_for_full_iqtree_run.output.model,
    output:
        done=temp(touch("{subdir}/reduced_alignments/{seq_id}/run_iqtree_restricted_alignments.done")),
        tree="{subdir}/reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile",
        mlfile="{subdir}/reduced_alignments/{seq_id}/reduced_alignment.fasta.iqtree",
        mldist="{subdir}/reduced_alignments/{seq_id}/reduced_alignment.fasta.mldist"
    shell:
        """
        iqtree -s {input.reduced_msa} -m $(cat {input.full_model}) --prefix {input.reduced_msa} -bb 1000 -redo
        """


rule extract_single_fastas:
    input:
        "{subdir}/convert_input_to_fasta.done",
        msa="{subdir}/"+input_alignment
    output:
        taxon_msa="{subdir}/reduced_alignments/{seq_id}/single_taxon.fasta",
        without_taxon_msa="{subdir}/reduced_alignments/{seq_id}/without_taxon.fasta"
    params:
        seq_id=lambda wildcards: wildcards.seq_id
    script:
        "scripts/extract_single_taxon_msa.py"


rule epa_reattachment:
    input:
        rules.run_iqtree_restricted_alignments.output.done,
        tree="{subdir}/reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile",
        model=rules.extract_model_for_full_iqtree_run.output.model,
        taxon_msa="{subdir}/reduced_alignments/{seq_id}/single_taxon.fasta",
        without_taxon_msa="{subdir}/reduced_alignments/{seq_id}/without_taxon.fasta",
    output:
        epa_result="{subdir}/reduced_alignments/{seq_id}/epa_result.jplace",
    params:
        output_folder="{subdir}/reduced_alignments/{seq_id}"
    shell:
        """
        model=$(cat {input.model})
        epa-ng --ref-msa {input.without_taxon_msa} --tree {input.tree} --query {input.taxon_msa} --model $model -w {params.output_folder} --redo
        """


rule write_reattached_trees:
    input:
        epa_result="{subdir}/reduced_alignments/{seq_id}/epa_result.jplace",
        restricted_trees="{subdir}/reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile",
    output:
        reattached_trees="{subdir}/reduced_alignments/{seq_id}/reattached_tree.nwk",
    script:
        "scripts/write_reattached_trees.py"


rule extract_reattachment_statistics:
    input:
        dynamic_input=dynamic_input,
        full_tree="{subdir}/"+input_alignment+".treefile",
        full_mldist_file="{subdir}/"+input_alignment+".mldist",
    output:
        plot_csv="{subdir}/reduced_alignments/reattachment_data_per_taxon_epa.csv",
        random_forest_csv="{subdir}/reduced_alignments/random_forest_input.csv",
        bootstrap_csv="{subdir}/reduced_alignments/bts_bootstrap.csv",
    script:
        "scripts/extract_reattachment_statistics.py"


rule random_forest_regression:
    input:
        csvs=expand("{subdir}/reduced_alignments/random_forest_input.csv", subdir=get_subdirs(data_folder)),
    output:
        model_features_file=data_folder+"model_feature_importances.csv",
        output_file_name=data_folder+"random_forest_regression.csv",
        combined_csv_path=data_folder+"combined_statistics.csv",
    params:
        column_to_predict = "normalised_tii",
        subdirs=get_subdirs(data_folder),
    script:
        "scripts/random_forest_regression.py"


rule random_forest_plots:
    input:
        random_forest_csv=rules.random_forest_regression.output.output_file_name,
        model_features_csv=rules.random_forest_regression.output.model_features_file,
        combined_csv_path=data_folder+"combined_statistics.csv",
    params:
        forest_plot_folder=data_folder+"plots/",
    output:
        temp(touch("random_forest_plots.done"))
    script:
        "scripts/random_forest_plots.py"


rule create_plots:
    input:
        csv="{subdir}/reduced_alignments/reattachment_data_per_taxon_epa.csv",
        bootstrap_csv="{subdir}/reduced_alignments/bts_bootstrap.csv",
    output:
        temp(touch("{subdir}/create_plots.done")),
    params:
        plots_folder="{subdir}/"+plots_folder,
    script:
        "scripts/create_plots.py"


rule create_other_plots:
    input:
        dynamic_input=dynamic_input,
        csv="{subdir}/reduced_alignments/reattachment_data_per_taxon_epa.csv",
        full_tree="{subdir}/"+input_alignment+".treefile",
        mldist="{subdir}/"+input_alignment+".mldist",
    output:
        temp(touch("{subdir}/create_other_plots.done")),
    params:
        plots_folder="{subdir}/"+plots_folder,
        subdir=lambda wildcards: wildcards.subdir,
    script:
        "scripts/create_other_plots.py"
