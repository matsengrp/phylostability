#!/bin/bash

basedir="$1"
directories=($(find -L "$basedir" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;))

output_file=$basedir/"nexus_data.csv"
if [[ -f $output_file ]]; then
    echo "nexus_data.csv already exists!"
    exit 0
fi
echo "file,taxa,sequences" > "$output_file"

# Enable case-insensitive pattern matching
shopt -s nocasematch

# Loop through Nexus files
for dir in "${directories[@]}"; do
    if [[ $dir =~ plots || $dir == selected_data || $dir == benchmarking ]]; then
        echo "Skippind directory $dir"
        continue
    fi
    echo "Processing directory $dir"
    for files in $basedir/$dir/*.nex; do
        for file in $files; do
            if [[ -f $file ]]; then
                if [[ $file == *.splits.nex ]]; then
                    echo "Skipping $file"
                    continue
                fi
                # echo $file
                # use regex to find number of taxa and number of sequences
                while IFS= read -r line; do
                    if [[ $line =~ ntax[a-zA-Z]*\s*=\s*([0-9]+) ]]; then
                        taxa="${BASH_REMATCH[1]}"
                    fi
                    if [[ $line =~ nchar[a-zA-Z]*\s*=\s*([0-9]+) ]]; then
                        sequences="${BASH_REMATCH[1]}"
                    fi
                done < "$file"

                if [[ -n $taxa && -n $sequences ]]; then
                    relative_filepath=${file#$basedir/}
                    # Append data to output file
                    echo "$relative_filepath,$taxa,$sequences" >> "$output_file"
                fi
            fi
        done
    done
done

