## Overview
This folder contains the scripts necessary to generate MP DAGs to compare to the result of the stability workflow.
It is not built into a pipeline like snakemake, but is organized into a series of scripts, each of which requires its own conda environment in order to be run.
The yml to generate each conda env is described in the corresponding script's header.

### Setting up parsimony DAGs to compare to ML output
In order to generate MP DAGs, let `MAIN_DIR` be the path to a completed run of the stability analysis. 
This directory will be something of the form  `/path/to/main/data/directory/selected_data/`.
Also needed is a local install of consel. 

Then run the following commands:
```
conda activate usher
./generate_usher_trees.sh MAIN_DIR full_alignment.fasta generate_usher_tree.snakemake

conda activate extended-historydag
./convert_usher_outputs_to_larch_inputs.sh MAIN_DIR larch_INPUT.fasta uncondensed-final-tree.nh larch_INPUT_DAG.pb

conda activate larch
./make_parsimony_dags.sh MAIN_DIR 80 larch_INPUT_DAG.pb larch_INPUT.vcf larch_OUTPUT_DAG.pb

conda activate extended-historydag
./compare_parsimony_dags_to_ML_trees.sh MAIN_DIR larch_OUTPUT_DAG.pb full_alignment.fasta.treefile larch_INPUT.fasta dag_comparsion_output.csv

./create_mds_plots.sh MAIN_DIR larch_OUTPUT_DAG.pb full_alignment.fasta.treefile 10000 larch_mds_plot.png

./compare_larch_modes_to_iqtree_results.sh MAIN_DIR larch_subset_newicks full_alignment.fasta.treefile 10000 comparison_mds.svg larch_INPUT.fasta iqtree-model.txt

./compare_iqtree_on_mds_clusters.sh MAIN_DIR full_alignment.fasta.iqtree iqtree-model.txt full_alignment.fasta.sitelh dag_comparison.csv dag_comparison_output.csv /path/to/consel/bin


```
This will create a csv in `MAIN_DIR`, each row of which corresponds to a dataset in the analysis, and  containing the following columns of data:
- `dataset` The name of the dataset in question
- `DAG parsimony score` The best parsimony score for any tree found by larch-usher
- `tree best parsimony score` The parsimony score obtained by running Fitch-Sankoff on the ML tree
- `one sided RF` The one-sided RF distance from the ML tree to the MP DAG. Since the IQTrees are bifurcating and the DAG allows multifurcations, the one-sided ML distance computes the number of splits in the DAG that are *not* in the ML tree.
- `min RF distance` The minimum (full) RF distance between any tree in the MP DAG and the ML tree
- `max RF distance` The maximum (full) RF distance between any tree in the MP DAG and the ML tree
- `num histories` The number of MP trees in the DAG
- `average pairwise RF distance` average RF distance between the MP DAG and the ML tree
- `reference sequence hamming distance` the hamming distance between the DAG's referenc sequence and the tree root's referenc sequence obtained by running Fitch-Sankoff.
