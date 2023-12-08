dynamic_input = snakemake.input.dynamic_input
output_file = snakemake.output.reattached_trees


num_taxa = int(len(dynamic_input)/4)
reattached_tree_paths = dynamic_input[3*num_taxa:]

with open(output_file, 'w') as outfile:
    for filename in reattached_tree_paths:
        try:
            with open(filename, 'r') as infile:
                first_line = infile.readline().strip()  # Read the first line
                outfile.write(first_line + '\n')  # Write the first line to the output file
        except IOError:
            print(f"Error: File {filename} not accessible")
