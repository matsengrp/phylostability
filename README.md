# README

<!-- TODO: Add link to paper -->
This repo provides a pipeline for analyzing the stability of taxa in all given alignments.
We add the option of first subsampling the given set of alignments to generate a dataset of alignment that contain different numbers of sequences and also vary in sequence length.


## Data

The directory containing the data on which the pipeline is to be run needs to be provided in `config.yaml` as *data_folder*.
This directory should contain subdirectories which themselves contain alignments as nexus or fasta files, or symlinks to such files.


### Subsampling datasets

When running the pipeline with `snakemake -cX`, *num_samples* alignments are chosen randomly so that different alignment lengths and number of taxa are included (aligmnents are binned according to those parameters and drawn uniformly from those bins).
If you want to run the analysis on all provided alignments, set *num_samples* in `config.yaml` to be the total number of datasets.
Alternatively, you could delete `selected_data/` from the data_folder path in `Analysis_snakefile` and run `snakemake --snakefile Analysis_snakefile -cX`.


## Prerequisites

Python packages that are required for running this pipeline can be found in `environment.yml`.
A conda environment called *phylostability* for our analysis can be installed using `conda env create -f environment.yml`.


## Running the pipeline

To run the workflow, execute:

`snakemake -c36`

Note that this will use *36* cores, you can adjust accordingly.
The number of cores to be used also needs to be updated in `config.yaml`.
The number of subsampled datasets is set in `config.yaml` as *num_samples* and defaults to *4*.
If the pipeline has run successfully, rerunning `snakemake -c36` will add *num_samples* to the already existing run and will perform random forest regression and classification on all datasets together.

The default *stability_measure* used in the analysis is the normalized Taxon Influence Index (*normalised_tii*), but this can be set to be *rf_radius* in `config.yaml`.


## Output

The names of datasets that are subsampled and used in the stability analysis are saved in the file `{data_folder}/selected_datasets.csv`, where *data_folder* is the name of the directory containing all data.
All data produced in the analysis is saved in the directory `{data_folder}/selected_data/`.
This includes a number of csv files as well as alignments and trees, which are saved in directories named by the names of the alignment files as provided in the input data.
The most important csv file is probably `{data_folder}/selected_data/rf_regression_combined_statistics.csv`, which contains all summary statistics computed for all taxa, which are then used for training random forests (or a balanced subset of taxa is used).
The following plots summarizing results of our stability analysis, including results of random forest regression and classification, and can be found in `{data_folder}/selected_data/plots`:

| Filename         | Plot     |
|--------------|-----------|
| au_pie_chart.pdf | Pie chart summarizing number of taxa for which AU-test is significant/non-significant and *stability_measure* is zero/non-zero      |
| au_test_classifier_features      | Feature importances of random forest classifier trained to predict AU-test output (significant vs non-significant)  |
| au_test_classifier_results.pdf | ROC curve for classifier predicting AU-test significance vs non-significance |
| bts_vs_bootstrap | Branch Taxon Score vs Bootstrap score of nodes incident to corresponding branch over all datasets|
| combined_random_forest_features.pdf | Feature importances of random forest classifier and regressor trained to predict *stability_measure* (classifier predicts TII zero vs non-zero) |
| discrete_random_forest_model_features.pdf | Feature importances of random forest classifier predicting *stability_measure* |
| normalised_tii.pdf | Histogram showing number of times each normalized TII value is observed |
| random_forest_classifier_results.pdf | ROC curve of random forest classifier predicting *stability_measure* zero vs non-zero |
| random_forest_model_features.pdf | Feature importances of random forest regression predicting *stability_measure* |
| random_forest_results.pdf | Scatterplot of predicted vs true values of *stability_measure* |
| rf_radius_num_taxa.pdf | Scatterplot of RF radius vs number of taxa of alignment of the taxon for which we observe that RF radius |
| rf_radius.pdf | Histogram showing number of times each normalized RF radius is observed |
| tii.pdf | Histogram showing number of times each TII is observed (not normalized) |