#!/bin/bash

# This script runs larch-usher on a set of inputs in a nested set of directories using SLURM.
#
# The code assumes a filestructure consisting of a main directory 'main_dir', 
# containing subdirectories, each of which contains a protobuf and (optionally) vcf suitable for larch input. If the provided name for the vcf file does not point to an existing file, then larch-usher is run without it. 
#
# The script calls 1 external script:
# - a script called run_larch_usher.sh
#   this script calls larch-usher with provided command-line arguments, suitable for SBATCH
#
# conda env: larch (see environment yml file at github.com/matsengrp/larch)

set -eu
maindir=$1 #e.g. /path/to/adaptive-evolution/selected_data/
iterations=$2 #e.g. 50
larchinputname=$3 #e.g. larch_INPUT.pb
vcfname=$4 #e.g. larch_INPUT.vcf
larchoutputname=$5 #e.g. larch_DAG.pb

jobnum=1
for sd in $maindir/*/; do
  if [ -f $sd/$larchinputname ]; then # check if there's a larch input file
    if ! [ -f $sd/$larchoutputname"_$iterations".pb ]; then # check if there's a larch output file already
      # now run larch on the specified input file
      logfilepath=$sd/larch-usher-log
      larch_usher_options=" -i $sd/$larchinputname -c $iterations -o $sd/"$larchoutputname"_"$iterations".pb --sample-method rf-maxsum"
      if [ -f $sd/$vcfname ]; then
        larch_usher_options= "$larch_usher_options -v $sd/$vcfname"
      fi
      sbatch -c 4 -J runlu$jobnum -o $logfilepath/lu$jobnum.log ./run_larch_usher.sh $larch_usher_options
      jobnum=$((jobnum+1))
    fi
  fi
done
