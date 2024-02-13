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


## Running the pipeline

To run the workflow, execute:

`snakemake -c36`

Note that this will use *36* cores, you can adjust accordingly.
The number of cores to be used also needs to be updated in `config.yaml`.
The number of subsampled datasets is set in `config.yaml` as *num_samples* and defaults to *4*.
If the pipeline has run successfully, rerunning `snakemake -c36` will add *num_samples* to the already existing run and will perform random forest regression and classification on all datasets together.