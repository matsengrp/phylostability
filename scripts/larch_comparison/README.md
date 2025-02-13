## Overview
This folder contains the scripts necessary to generate MP DAGs to compare to the result of the stability workflow.
The workflow it executes should only be run on the completed output of the main `Snakefile`.

Since this analysis relies heavily on slurm, it is not built into a pipeline like snakemake, but is organized into a series of scripts, each of which requires its own conda environment in order to be run.

### Setting up parsimony DAGs to compare to ML output
In order to generate MP DAGs, let `MAIN_DIR` be the path to a completed run of the stability analysis. 
This directory will be something of the form  `/path/to/main/data/directory/selected_data/`.
Also needed is a download of faToVcf (see instructions [here](https://usher-wiki.readthedocs.io/en/latest/Installation.html)), a local install of [usher](https://github.com/yatisht/usher/) and of [larch](https://github.com/matsengrp/larch/), each with their associated conda environments.

Then run the following commands:
```
conda activate usher
./generate_usher_trees.sh MAIN_DIR full_alignment.fasta generate_usher_tree.snakemake

conda activate extended-historydag
./convert_usher_outputs_to_larch_inputs.sh MAIN_DIR larch_INPUT.fasta uncondensed-final-tree.nh larch_INPUT_DAG.pb

conda activate larch
./make_parsimony_dags.sh MAIN_DIR 80 larch_INPUT_DAG.pb larch_INPUT.vcf larch_OUTPUT_DAG.pb
```
Notes:
- The script `make_parsimony_dags.sh` submits jobs using SLURM, and so it is necessary to wait until all of these jobs have been completed before running a comparison between the MP output and the ML output.
- The script `run_larch_usher.sh` is used to submit the SLURM jobs, and contains a path to the local build of larch. This should be edited accordingly.

### Comparing ML output to parsinomy DAGs
Also needed is a local install of consel.

The output of `make_parsimony_dags.sh` is an untrimmed DAG in each subdirectory, titled `larch_OUTPUT_DAG.pb`.
In order to extract the MP trees in that DAG, visualize the space of trees in the trimmed DAG, and compare the ML tree to trees in that space, run the following set of commands:
```
conda activate extended-historydag
./compare_parsimony_dags_to_ML_trees.sh MAIN_DIR larch_OUTPUT_DAG.pb full_alignment.fasta.treefile larch_INPUT.fasta dag_comparsion_output.csv

# NOTE that the following script submits jobs to the cluster using SLURM. In order to insure all commands are run, rerun this script once again after the jobs are complete.
./create_mds_plots.sh MAIN_DIR larch_OUTPUT_DAG.pb full_alignment.fasta.treefile 10000 larch_mds_plot.png

./compare_larch_modes_to_iqtree_results.sh MAIN_DIR larch_subset_newicks full_alignment.fasta.treefile 10000 comparison_mds.svg larch_INPUT.fasta iqtree-model.txt

./compare_iqtree_on_mds_clusters.sh MAIN_DIR full_alignment.fasta.iqtree iqtree-model.txt full_alignment.fasta.sitelh dag_comparison_output.csv dag_comparison_output.csv /path/to/consel/bin
```

### Outputs from comparison
There are 3 main components to this comparison: 

1. Comparison between the MP DAG and the ML tree: 
  - Is the ML tree's topology found in the MP DAG?
  - How close is the ML tree to the nearest MP tree in the DAG?
  - How does the best parsimony score obtainable on the MP tree compare to the MP DAG?
2. Exploration of the space of MP trees in the DAG: 
  - How many MP trees are there in the DAG?
  - Is the space topologically diverse? 
  - are there well-defined clusters in the space of trees? How many, if so?
3. Comparison of ML tree with representative trees from the clusters found in the MP DAG tree-space
  - Is the ML tree in one of the clusters?
  - Is there a significant difference in the ML tree and any of the clusters?
  - Is the best model found for the ML tree also optimal for a representative for each cluster?

The scripts that generate an MP DAG also create a csv in `MAIN_DIR`, each row of which corresponds to a dataset in the analysis, and  containing the following columns of data:
- `dataset` The name of the dataset in question
- `DAG parsimony score` The best parsimony score for any tree found by larch-usher
- `tree best parsimony score` The parsimony score obtained by running Fitch-Sankoff on the ML tree
- `one sided RF` The one-sided RF distance from the ML tree to the MP DAG. Since the IQTrees are bifurcating and the DAG allows multifurcations, the one-sided ML distance computes the number of splits in the DAG that are *not* in the ML tree.
- `min RF distance` The minimum (full) RF distance between any tree in the MP DAG and the ML tree
- `max RF distance` The maximum (full) RF distance between any tree in the MP DAG and the ML tree
- `num histories` The number of MP trees in the DAG
- `average pairwise RF distance` average RF distance between the MP DAG and the ML tree
- `reference sequence hamming distance` the hamming distance between the DAG's referenc sequence and the tree root's referenc sequence obtained by running Fitch-Sankoff.
These are used to answer the questions in 1 above. 

The scripts that calculate and plot an MDS matrix for the subset of trees in the DAG also generate the following outputs:
- A subdirectory called `larch_dm/` inside each dataset's specific directory.
- A 

The scripts that compare the representatives from each cluster generate the following output:
- A csv in `MAIN_DIR` that contains the columns from the MP DAG comparison of part 2, but with the added columns: 
  - `num_clusters` the number of clusters found for the space of MP trees
  - `num_models` the number of unique models that IQTree's ModelFinder identified for the set of cluster-centroids and the ML tree combined.
  - `MP-topology au-test p-scores` the p-score from performing the AU test on each of the clusters' centroid topology to test whether the ML tree is significantly better.
  - `inverse au-test p-scores` the p-score from performing the AU test on each of the clusters' centroid topology to test whether the centroid topology is significantly better than the ML tree.
  - `MP-topology log-likelihoods` the log-likelihoods of the cluster centroid topologies.


### breakdown of scripts by usage
Generating MP DAG:
 - `convert_usher_outputs_to_larch_inputs.sh` iterates over the subdirectories corresponding to each dataset and calls the script `create_pb_dag_from_newick_and_fasta` to generate a starting tree in that directory for the MP DAG search.
- `create_pb_dag_from_newick_and_fasta.py` uses the historydag package in python to convert a newick tree to a protobuf, which can be loaded into larch. 
- `generate_usher_tree.snakemake` this snakefile creates a starting tree topology in newick format and a vcf. This script uses `usher` to generate the starting tree and faToVcf to create the vcf.
- `generate_usher_trees.sh` this script iterates over the subdirectories corresponding to each dataset and calls the snakefile `generate_usher_tree.snakefile` on each.
- `make_parsimony_dags.sh` this script iterates over each dataset's directory and submits a SLURM job to run larch and generate a MP DAG from the initial starting tree.

Comparing all trees in MP DAG to ML tree:
- `compare_dag_to_trees.py` this script computes the RF distances between the ML tree and the MP DAG, also the best parsimony score of each, and the number of histories in the MP DAG.
- `compare_parsimony_dags_to_ML_trees.sh` this script iterates over all datasets and calls the script `compare_dag_to_trees.py` to compare the ML tree found by IQTree and the MP DAG generated by larch in each directory.

Visualizing space of MP trees:
- `combine_indexed_dms.py` helper function for generating the distance matrix for a subset of trees in the MP DAG. It combines the 100x100 blocks of the DM matrix that are each computed separately in parallel. 
- `compute_mds_matrix.py` this script iterates over all of the datasets, creates a subset of the MP trees, generates the pairwise distance matrix for those trees, and creates an MDS plot of those.
- `dm_block.sh` helper function for generating the distance matrix. This script populates a 100x100 block of this matrix.
- `extract_newicks_subset.py` generates a random subset of the trees in the MP DAG
- `full_rf_distance_matrix.py` computes the RF distance between two trees.
- `make_dm.sh` helper function for the distance matrix. This script creates a file for each 100x100 block of the matrix, and calls `dm_block.sh` on these files.
- `make_dm_file.py` helper function for generating the distance matrix. This script creates a numpy memmap file of the appropriate type and dimension for reading/writing.
- `plot_mds.py` plots the MDS scaling from the distance matrix.

Comparing representatives of MP-space to ML tree:
- `compare_larch_modes_to_iqtree_results.py` performs k-means clustering on the MDS scaling of MP trees. This script uses a combined silhouette and elbow analysis to identify the optimal number of clusters, performs the clustering, and extracts a topology that is closest to each cluster's centroid. It then runs IQTree constrained to the topology of each of these centroids to compare with the original ML tree.
- `compare_larch_modes_to_iqtree_results.sh` iterates over all datasets and calls the python script on it.

General utility:
- `call_python_on_script.sh` provides a path to a local conda build (that contains the environments listed in this README)
- `run_larch_usher.sh` provides a path to a local build of larch
- `resolve_inputs.py` this script reads a newick file and a fasta, and writes a new fasta file whose keys are the subset of the original fasta that correspond to leaves of the newick tree.

