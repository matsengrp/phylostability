#!/bin/bash

basedir="$1"
directories=($(find -L "$basedir" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;))

output_file=$basedir/"nexus_data.csv"
echo "file,taxa,sequences" > "$output_file"

# Enable case-insensitive pattern matching
shopt -s nocasematch

# Loop through Nexus files
for dir in "${directories[@]}"; do
    if [[ $dir =~ plots ]]; then
        echo "Skippind directory $dir"
        continue
    fi
    echo "Processing directory $dir"
    for files in $basedir/$dir/*.nex; do
        echo $files
        for file in $files; do
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
        done
    done
done

