from ete3 import Tree

tree_nwk = snakemake.input[1]
output_file = snakemake.output[0]
seq_id = snakemake.params[0]


# open the newick and save the topology
with open(tree_nwk, "r") as f:
    nh_string = f.readlines()[0].strip()
reduced_topology = Tree(nh_string)

# for each node(considered as the child of its parent edge), create a copy of
# the original topology and attach the pruned taxon as a sister of that node
# (i.e. attach the pruned taxon at that edge), and write new topology to file
lookup_val = 0
for node in reduced_topology.traverse("postorder"):
    if not node.is_root():
        node.add_features(lookup_key=str(lookup_val))
        augmented_topology = reduced_topology.copy(method="deepcopy")
        sibling = augmented_topology.search_nodes(lookup_key=str(lookup_val))[0]
        sibling.add_sister(name=seq_id)
        augmented_topology.write(format=1, outfile=output_file)
        lookup_val += 1
