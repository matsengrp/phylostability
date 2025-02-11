#!/bin/bash

file1=$1
file2=$2
file1_tree_line=$3
file2_tree_line=$4
file1_num_trees=$5
file2_num_trees=$6
dm_file=$7
file1_startend_idx=$8
file2_startend_idx=$9
one_sided=0
imin=$( cut -d ',' -f 1 <<< $file1_startend_idx )
imax=$( cut -d ',' -f 2 <<< $file1_startend_idx )
jmin=$( cut -d ',' -f 1 <<< $file2_startend_idx )
jmax=$( cut -d ',' -f 2 <<< $file2_startend_idx )
file1_max=$((imax - imin))
file2_max=$((jmax - jmin))
echo "filemaxes: $file1_max, $file2_max"

if [[ $file1 != $file2 ]]; then
  one_sided=1
fi
file1_num_lines=$( wc -l < $file1 )
file2_num_lines=$( wc -l < $file2 )
file1_beg_lines=$( cat $file1 | grep $file1_tree_line --line-number | head -n $((imin + 1)) | tail -n 1 | cut -d ':' -f 1 )
file2_beg_lines=$( cat $file2 | grep $file2_tree_line --line-number | head -n $((jmin + 1)) | tail -n 1 | cut -d ':' -f 1 )

iii=0
ict=$imin
for ((ii=$file1_beg_lines; ii<= $file1_num_lines; ii++)); do
  jct=$jmin
  if [ "$ict" -lt "$imax" ]; then
    tree1line=$( sed -n $ii"p" $file1 )
    if [[ "$tree1line" =~ "$file1_tree_line" ]]; then
      ict=$((ict + 1))
      line1="(""$( cut -d '(' -f 2- <<< $tree1line )"
      jjmin=$file2_beg_lines
      if [ $one_sided == 0 ]; then
        if [ "$jjmin" -le "$file1_beg_lines" ]; then
          jjmin=$((ii + 1))
          jct=$ict
        fi
      fi
      jjj=0
      for ((jj=$jjmin; jj<= $file2_num_lines; jj++)); do
        if [ "$jct" -lt "$jmax" ]; then
          tree2line=$( sed -n $jj"p" $file2 )
          if [[ "$tree2line" =~ "$file2_tree_line" ]]; then
            jct=$((jct + 1))
            line2="(""$( cut -d '(' -f 2- <<< $tree2line )"
            line1=$( tr -d '\n\t\r ' <<< "$line1" )
            line2=$( tr -d '\n\t\r ' <<< "$line2" )
            ./call_python_on_script.sh full_rf_distance_matrix.py $dm_file $iii $jjj $line1 $line2 $file1_max $file2_max
            jjj=$((jjj+1))
          fi
        fi
      done
      iii=$((iii+1))
    fi
  fi
done;

dm_done_file="${dm_file%_${dm_file##*_}}"
dm_done_file="${dm_done_file%_${dm_done_file##*_}}_done"
echo "$imin $jmin" >> $dm_done_file
