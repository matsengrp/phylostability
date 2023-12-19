# README

## Subsample datasets

To run the workflow on a subsample of the Harrington et al. datasets, execute:

`snakemake -c36`

Note that this willl use *36* cores, you can adjust accordingly.
The number of subsampled datasets is set in `config.yaml` as *num_samples* and is currently set to *1* (for testing).
If the whole process has been run successfully, the next one will add *num_samples* to the already existing run and will perform random forest regression and classification on all datasets together.

## Data
The data on which the pipeline should be run needs to be stored in a folder that is defined in `config.yaml` as data_folder.
This folder should contain subfolders which themselves contain alignments as nexus or fasta files, or symlinks to such files.