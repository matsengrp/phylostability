#!/bin/bash

data_dir=$1 # e.g. "/path/to/harrington_data/selected_data"
alignment_name=$2 # e.g. "full_alignment.fasta"
which_raxml=$3 # e.g. /path/to/raxml-ng
excluded_dirs="benchmarking normalised_tii plots rf_radius"

for dir in $data_dir/*/; do
  latest_dir="${dir%/}"
  latest_dir="${latest_dir##*/}"
  listfound=$(echo $excluded_dirs | grep -w -q $latest_dir)
  if [ "$listfound" == "" ]; then
    if [ -f $dir/$alignment_name ]; then
      echo "Alignment:"
      echo $dir/$alignment_name
      if ! [ -f $dir/pythia_difficulty.txt ]; then
        echo "$(date): $dir"
        pythia -m $dir/$alignment_name -r $which_raxml -o $dir/pythia_difficulty.txt
      fi
    fi
  fi
  touch $data_dir/"run_pythia.done"
done
