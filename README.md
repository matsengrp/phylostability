# README

## Subsample datasets

To run the workflow on a subsample of the Harrington et al. datasets, execute:

`snakemake --snakefile Subsampling_snakefile -c8`

Note that this willl use 8 cores, you can adjust accordingly.
Also, if you add `-R select_datasets`, it will add `num_samples` datasets to the selected samples that already are in `harrington_data/selected_data/`, where `num_samples` is set in the beginning of `Subsampling_snakefile`.