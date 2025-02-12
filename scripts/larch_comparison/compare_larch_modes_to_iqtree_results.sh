#!/bin/bash

# environments: 
# - extended-historydag
#   **env file** is at github.com:matsengrp/phylostability/scripts/larch_comparison/extended_historydag_env.yml
#   **make sure** build sequence for github.com:matsengrp/historydag has been run in this environment
# - phylostability
#   **env file** is at github.com:matsengrp/phylostability/environment.yml

set -eu
eval "$(conda shell.bash hook)"
maindir=$1 #e.g. /path/to/adaptive-evolution/selected_data/
larchnewicks=$2 #e.g. larch_subset_newicks
treefile=$3 #e.g. full_alignment.fasta.treefile
numtrees=$4 #e.g. 10000
outputfile=$5 #e.g. comparison_mds.svg
fastafile=$6 #e.g. larch_INPUT.fasta
oldmodel=$7 #e.g. iqtree-model.txt

clusterplotfilename="comparison_mds_with_clustering.svg"

for sd in $maindir/*/; do
  if [[ -f $sd/$larchnewicks && -f $sd/$treefile && -f $sd/$outputfile && -f $sd/$fastafile ]]; then
    if [[ -f $sd/larch_dm/mds && -f $sd/larch_dm/dm_dim ]]; then
      conda activate historydag
      if ! [ -f $sd/larch_dm/cluster_centroids_nwks ]; then
        echo "creating MDS clusters for $sd"
        mds_dim=$( cat $sd/larch_dm/dm_dim | awk '{print $1}' )
        if [ -f $sd/larch_dm/best_k_for_clustering ]; then
          python compare_larch_modes_to_iqtree_results.py $sd/larch_dm/mds $mds_dim $sd/larch_dm/cluster_labels $sd/larch_dm/cluster_centroids_nwks $sd/larch_dm/cluster_centroids_idxs $sd/larch_subset_newicks $sd/larch_dm/best_k_for_clustering
        else
          python compare_larch_modes_to_iqtree_results.py $sd/larch_dm/mds $mds_dim $sd/larch_dm/cluster_labels $sd/larch_dm/cluster_centroids_nwks $sd/larch_dm/cluster_centroids_idxs $sd/larch_subset_newicks
        fi

      fi
      if ! [ -f $sd/$clusterplotfilename ]; then
        echo "plotting MDS clusters for $sd"
        python plot_mds.py $sd/larch_dm/mds $sd/larch_dm/dm_dim $sd/larch_dm/num_larch_trees $sd/$clusterplotfilename $sd/larch_dm/cluster_labels_centroid_annotated
      fi

      if [[ -f $sd/larch_dm/cluster_centroids_nwks && -f $sd/larch_dm/cluster_centroids_idxs ]]; then
        conda activate phylostability
        numclusters=$( wc -l $sd/larch_dm/cluster_centroids_idxs | awk '{print $1}' )
        echo "extracting topological mean for $numclusters clusters and running iqtree"
        rm -rf $sd/larch_dm/iqtree_cluster_comparison/
        mkdir -p $sd/larch_dm/iqtree_cluster_comparison
        for cluster in `seq 1 1 $numclusters`; do
          nwkline=$((2*cluster - 1))
          echo $(sed -n $nwkline'p' < "$sd/larch_dm/cluster_centroids_nwks" ) > "$sd/larch_dm/iqtree_cluster_comparison/cluster_$cluster"
          # find best model for this topology/msa
          iqtree -g "$sd/larch_dm/iqtree_cluster_comparison/cluster_$cluster" -s $sd/$fastafile --prefix "$sd/larch_dm/iqtree_cluster_comparison/cluster_$cluster""_model" -m MF -redo

          # run iqtree with specified model & topology
          found_model=$(grep "Model of substitution" -i "$sd/larch_dm/iqtree_cluster_comparison/cluster_$cluster""_model.iqtree")
          found_model=${found_model:23}
          iqtree -g "$sd/larch_dm/iqtree_cluster_comparison/cluster_$cluster" -s $sd/$fastafile --prefix "$sd/larch_dm/iqtree_cluster_comparison/cluster_$cluster" -wsl -m "$found_model" -bb 1000 -redo
          old_model="$(cat $sd/$oldmodel)"
          if [[ $old_model != $found_model ]]; then
            iqtree -g "$sd/larch_dm/iqtree_cluster_comparison/cluster_$cluster" -s $sd/$fastafile --prefix "$sd/larch_dm/iqtree_cluster_comparison/cluster_$cluster""_oldmodel" -wsl -m "$old_model" -bb 1000 -redo
          fi
        done
      fi
    fi
  fi
done
