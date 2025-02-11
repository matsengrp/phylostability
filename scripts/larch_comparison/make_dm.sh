#!/bin/bash
fn1=$1
fn2=$2
fn1max=$3
fn2max=$4
treeline1=$5
treeline2=$6
outfname=$7
logfolder=$8

step=100
jmin=0
fn1maxm1=$((fn1max - 1))
fn2maxm1=$((fn2max - 1))

if ! [ -f $outfname ]; then
  ./call_python_on_script.sh make_dm_file.py $outfname $fn1max $fn2max
fi
for i in `seq 0 $step $fn1maxm1`; do
  currenti=$((i + $step))
  if [ $fn1 == $fn2 ]; then
    jmin=$i
  fi
  for j in `seq $jmin $step $fn2maxm1`; do

   if ! [ -f $outfname"_"$i"_"$j ]; then
     ./call_python_on_script.sh make_dm_file.py $outfname"_"$i"_"$j $step $step
   fi
    currentj=$((j + $step))
    sbatch -c 12 -J mat$i-$j -o $logfolder/mat$i"_"$j"_log" dm_block.sh $fn1 $fn2 $treeline1 $treeline2 $fn1max $fn2max $outfname"_"$i"_"$j $i,$currenti $j,$currentj
    echo "submitted $step x $step block starting at $i, $j at $(date) for $outfname"
    # sleep 5m
  done
done
