dynamic_input = snakemake.input.dynamic_input
full_treefile = snakemake.input.full_tree
output_file = snakemake.output.reattached_trees
order_file = snakemake.output.reattached_trees_order


num_taxa = int(len(dynamic_input) / 4)
reattached_tree_paths = dynamic_input[3 * num_taxa :]
order = []

with open(output_file, "w") as outfile:
    with open(full_treefile, "r") as infile:
        outfile.write(infile.readline().strip() + "\n")
        order.append("full")
    for filename in reattached_tree_paths:
        try:
            with open(filename, "r") as infile:
                first_line = infile.readline().strip()  # Read the first line
                outfile.write(
                    first_line + "\n"
                )  # Write the first line to the output file
                order.append(filename.split("/")[-2])
        except IOError:
            print(f"Error: File {filename} not accessible")

with open(order_file, "w") as file:
    for seq_id in order:
        file.write(seq_id + "\n")