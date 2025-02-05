#!/bin/bash

# conda env: historydag (see environment yml file at github.com/matsengrp/historydag)
set -eu
maindir=$1        #e.g. /path/to/adaptive-evolution/selected_data/
larchtreefile=$2  #e.g. larch_DAG.pb
mltreefile=$3     #e.g. full_alignment.fasta.treefile
inputfastafile=$4 #e.g. larch_INPUT.fasta
outputfile=$5     #e.g. dag_comparison_output.csv

for sd in $maindir/*/; do
  if [ -f $sd/$mltreefile ]; then
    if [ -f $sd/$larchtreefile ]; then
      echo "comparing $sd"
      python compare_dag_to_trees.py $sd/$inputfastafile $sd/$mltreefile $sd/$larchtreefile $maindir/$outputfile $sd/dag_comparison.png
    fi
  fi
done

