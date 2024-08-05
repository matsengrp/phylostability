#!/bin/bash

data_dir=$1 # e.g. "/path/to/harrington_data/selected_data"
which_raxml=$2
excluded_dirs="benchmarking normalised_tii plots rf_radius"

for dir in $data_dir/*/; do
  latest_dir="${dir%/}"
  latest_dir="${latest_dir##*/}"
  echo "$latest_dir: $listfound"
  listfound=$(echo $excluded_dirs | grep -w -q $latest_dir)
  if [ "$listfound" == "" ]; then
    if [ -f $dir/full_alignment.fasta ]; then
      pythia -m $dir/full_alignment.fasta -r $which_raxml -o $dir/pythia_difficulty.txt
    fi
  fi
done
