# README

## Subsample datasets

To extract the number of taxa and alignments length from a directory, execute:

`./scripts/get_nexus_data.sh DIRECTORY`

 where `DIRECTORY` is the name of a directory that contains sub-directories containing alignments as fasta or nexus files.

 To then create a folder `DIRECTORY/selected_data` that contains `N` datasets, run

 `python scripts/select_datasets.py DIRECTORY N`

 where `DIRECTORY` is the same directory name as above and `N` is the number of subsampled datasets we want.

 ## Run Snakemake

 Note that to run snakemake on the subsampled dataset, the `data_folder` parameter in the beginning of the Snakefile needs to be set to
 `data_folder = DIRECTORY/selected_data/`