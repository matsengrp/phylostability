#!/bin/bash

# This script produces inputs(a non-optimized tree from usher and a vcf file)
# suitable as a starting point for larch-usher runs in a nested set of directores. 
#
# The code assumes a filestructure consisting of a main directory 'main_dir', 
# containing subdirectories, each of which contains a fasta and iqtree output file
#
# The script calls 2 external executables:
# - a python script called "phylostability/scripts/resolve_inputs.py"
#   this script cleans up the fasta file in each directory
# - a snakefile that actually converts the input fasta into the output.
#   examples of such snakefiles are in /fh/fast/matsen_e/mbarker/larch/setup_larch_inputs
#
# conda env: usher (see /fh/fast/matsen_e/mbarker/larch/setup_larch_inputs/ yml file)

set -eu
maindir=$1 #e.g. adaptive-evolution/selected_data/
fastafilename=$2 # e.g. full_alignment.fasta
setuplarchscript=$3 # e.g. /fh/fast/matsen_e/mbarker/larch/setup_larch_inputs/convert_fasta_and_newick_to_larch_input_with_specified_filenames.snakefile
treefilename="$fastafilename.treefile"
larchinputname="larch_INPUT"

for sd in $maindir/*/; do
  echo "looking at $sd"
  if [[ -f $sd/$treefilename ]]; then # check if there's an iqtree output
    if [[ -f $sd/$fastafilename ]]; then # check if there's a fasta file
      if ! [ -f $sd/$larchinputname.pb ]; then # check if there's already a larch input file
        # clean up the fasta file and give it a new special name
        python resolve_inputs.py $sd/$treefilename $sd/$fastafilename $sd/$larchinputname"_setup.fasta"
        # create a custom config file for the snakefile based on the current subdirectory
        echo 'fasta_filename: "'$larchinputname'_setup.fasta"' > $sd/config.yaml
        echo 'newick_filename: "'$treefilename'"' >> $sd/config.yaml
        echo 'base_filename: "'$larchinputname'"' >> $sd/config.yaml
        # call the workflow to set up larch inputs based on the fasta and config
        snakemake --snakefile $setuplarchscript -d $sd --use-conda -c1
      fi
    fi
  fi
done

