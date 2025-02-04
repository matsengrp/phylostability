#!/bin/bash

# This script converts the newick/fasta combination from the script 'generate-usher-trees.sh' into a DAG protobuf file. It is needed because the protobuf tree that usher generates is often condensed, and so its leafset does not match the leafsets of the iqtree outputs.
#
# The code assumes a filestructure consisting of a main directory 'main_dir', 
# containing subdirectories, each of which contains a newick and vcf suitable for larch input.
#
# The script calls 1 external script:
# - a python script called "phylostability/scripts/create_pb_dag_from_newick_and_fasta.py"
#
# conda env: historydag (see environment yml file at github.com/matsengrp/historydag)

set -eu
maindir=$1 # e.g. /path/to/adaptive-evolution/selected_data/
fastafilename=$2 # e.g. full_alignment.fasta
newickfilename=$3 # e.g uncondensed-final-tree.nh
outputfilename=$4 # e.g larch_INPUT.pb

for sd in $maindir/*/; do
  if [ -f $sd/$newickfilename ]; then
    if [ -f $sd/$fastafilename ]; then
      echo "creating input for $sd"
      python create_pb_dag_from_newick_and_fasta.py $sd/$fastafilename $sd/$newickfilename $sd/$outputfilename
    fi
  fi
done

