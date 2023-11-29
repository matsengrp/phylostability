from Bio import SeqIO
import os
import glob

# input/output file names
input_alignment="full_alignment.fasta"
data_folder="data/"
plots_folder="plots/"
IQTREE_SUFFIXES=["iqtree", "log", "treefile", "ckp.gz"]

subdirs = [f.path for f in os.scandir(data_folder) if f.is_dir() and ("plot" not in f.path and "benchmarking" not in f.path)]

# Retrieve all sequence IDs from the input multiple sequence alignment
def get_seq_ids(input_file, filetype):
    return [record.id for record in SeqIO.parse(input_file, filetype)]

seq_ids = {}
for subdir in subdirs:
    nexus_files = glob.glob(os.path.join(subdir, "*.nex"))
    fasta_files = glob.glob(os.path.join(subdir, "*.fasta"))
    nexus_files = [os.readlink(file) if os.path.islink(file) else file for file in nexus_files]
    fasta_files = [os.readlink(file) if os.path.islink(file) else file for file in fasta_files]

    if len(fasta_files) > 0:
        seq_ids[subdir] = get_seq_ids(fasta_files[0], "fasta")
    elif len(nexus_files) > 0:
        seq_ids[subdir] = get_seq_ids(nexus_files[0], "nexus")
    else:
        raise Exception("Error in subdirectory " + subdir + ": no input file\n")



# Define the workflow
rule all:
    input:
        "random_forest_plots.done",
        "benchmarking_plots.done",
        expand("{subdir}/create_plots.done", subdir=subdirs),
        expand("{subdir}/create_other_plots.done", subdir=subdirs),


# convert input alignments from nexus to fasta, if necessary
rule convert_input_to_fasta:
    input:
        data_folder=data_folder,
    output:
        temp(touch("convert_input_to_fasta.done")),
    params:
        subdirs=subdirs,
        fn=input_alignment
    script:
        "scripts/convert_input_to_fasta.py"


# Define the rule to extract the best model for iqtree on the full MSA
rule model_test_iqtree:
    input:
        "convert_input_to_fasta.done",
        msa="{subdir}/"+input_alignment
    output:
        temp(touch("{subdir}/model-test-iqtree.done")),
        modeltest="{subdir}/"+input_alignment+"_model.iqtree"
    benchmark:
        touch("{subdir}/benchmarking/benchmark_model_test_iqtree.txt")
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
        "convert_input_to_fasta.done",
        msa="{subdir}/"+input_alignment
    output:
        reduced_msa=temp("{subdir}/reduced_alignments/{seq_id}/reduced_alignment.fasta"),
    benchmark:
        touch("{subdir}/benchmarking/benchmark_remove_sequence_{seq_id}.txt")
    params:
        seq_id=lambda wildcards: wildcards.seq_id
    script:
        "scripts/remove_sequence.py"


# Define the rule to run IQ-TREE on the full MSA and get model parameters
rule run_iqtree_on_full_dataset:
    input:
        "convert_input_to_fasta.done",
        msa="{subdir}/"+input_alignment,
        full_model="{subdir}/iqtree-model.txt"
    output:
        temp(touch("{subdir}/run_iqtree_on_full_dataset.done")),
        tree="{subdir}/"+input_alignment+".treefile",
        mldist="{subdir}/"+input_alignment+".mldist"
    benchmark:
        touch("{subdir}/benchmarking/benchmark_run_iqtree_on_full_dataset.txt")
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
    benchmark:
        touch("{subdir}/benchmarking/benchmark_run_iqtree_restricted_alignments_{seq_id}.txt")
    shell:
        """
        iqtree -s {input.reduced_msa} -m $(cat {input.full_model}) --prefix {input.reduced_msa} -bb 1000 -redo
        """


rule extract_single_fastas:
    input:
        "convert_input_to_fasta.done",
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
    benchmark:
        touch("{subdir}/benchmarking/benchmark_epa_reattachment_{seq_id}.txt")
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


# paths containing subdirs and seq_ids for rule
# extract_reattachment_statistics
epa_paths = []
restricted_trees = []
reduced_mldists = []
for subdir in subdirs:
    for seq_id in seq_ids.get(subdir, []):
        epa_paths.append(f"{subdir}/reduced_alignments/{seq_id}/epa_result.jplace")
        restricted_trees.append(f"{subdir}/reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile")
        reduced_mldists.append(f"{subdir}/reduced_alignments/{seq_id}/reduced_alignment.fasta.mldist")


rule extract_reattachment_statistics:
    input:
        epa_results=epa_paths,
        restricted_trees=restricted_trees,
        full_tree=expand("{subdir}/"+input_alignment+".treefile", subdir=subdirs),
        full_mldist_file=expand("{subdir}/"+input_alignment+".mldist", subdir=subdirs),
        restricted_mldist_files=reduced_mldists,
    output:
        plot_csv=expand("{subdir}/reduced_alignments/reattachment_data_per_taxon_epa.csv", subdir=subdirs),
        random_forest_csv=expand("{subdir}/reduced_alignments/random_forest_input.csv", subdir=subdirs),
        bootstrap_csv=expand("{subdir}/reduced_alignments/bts_bootstrap.csv", subdir=subdirs)
    params:
        subdirs=subdirs
    script:
        "scripts/extract_reattachment_statistics.py"


rule random_forest_regression:
    input:
        csvs=expand("{subdir}/reduced_alignments/random_forest_input.csv", subdir=subdirs),
    output:
        model_features_file="model_feature_importances.csv",
        output_file_name=data_folder+"random_forest_regression.csv",
        combined_csv_path=data_folder+"combined_statistics.csv",
    benchmark:
        touch(data_folder + "benchmarking/benchmark_random_forest_regression.txt")
    params:
        column_to_predict = "normalised_tii",
        subdirs=subdirs,
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


reattached_trees = []
for subdir in subdirs:
    for seq_id in seq_ids.get(subdir, []):
        restricted_trees.append(f"{subdir}/reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile")
        reattached_trees.append(f"{subdir}/reduced_alignments/{seq_id}/reattached_tree.nwk")


rule create_other_plots:
    input:
        csv="{subdir}/reduced_alignments/reattachment_data_per_taxon_epa.csv",
        full_tree="{subdir}/"+input_alignment+".treefile",
        reduced_trees=restricted_trees,
        reattached_trees=reattached_trees,
        mldist="{subdir}/"+input_alignment+".mldist",
        reduced_mldist=reduced_mldists,
    output:
        temp(touch("{subdir}/create_other_plots.done")),
    params:
        plots_folder="{subdir}/"+plots_folder,
        subdir=lambda wildcards: wildcards.subdir,
    script:
        "scripts/create_other_plots.py"


# create single file for each dataset with timing breakdowns for the rules
rule combine_benchmark_outputs:
    input:
        "{subdir}/create_plots.done",
        sequence_ids=seq_ids.get("{subdir}", [])
    output:
        output_file="{subdir}/benchmarking_data.csv"
    params:
        subdir=lambda wildcards: wildcards.subdir,
        benchmarking_folder="{subdir}/benchmarking/",
        output_plot_path="{subdir}/"+plots_folder
    script:
        "scripts/combine_benchmark_outputs.py"


rule benchmark_outputs_across_datasets:
    input:
        dfs=expand("{subdir}/benchmarking_data.csv", subdir=subdirs),
        fastas=expand("{subdir}/" + input_alignment, subdir=subdirs)
    output:
        temp(touch("benchmarking_plots.done"))
    params:
        plots_folder = data_folder + "plots/"
    script:
        "scripts/benchmark_outputs_across_datasets.py"
