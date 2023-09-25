import os
from Bio import SeqIO
from ete3 import Tree, PhyloTree


# input/output file names
input_alignment="input_alignment.fasta"
output_folder="data/"

 
# Retrieve all sequence IDs from the input multiple sequence alignment
def get_seq_ids(input_file):
    return [record.id for record in SeqIO.parse(input_file, "fasta")]


# extract a dictionary of branch lengths keyed on the child node name of the corresponding branch
def get_branch_lengths(input_tree_file):
    with open(input_tree_file, "r") as f:
        tree_nwk = f.readlines()[0]
    input_tree = Tree(tree_nwk)
    lengths={}
    ctr = 0
    for node in input_tree.traverse("postorder"):
        if len(node.get_ancestors()) > 0:
            node_str = ",".join(sorted(list([l.name for l in node.get_leaves()])))
            lengths[str(ctr)] = [node_str, node.get_ancestors()[0].get_distance(node)]
        ctr += 1
    return lengths


# find the line in an iqtree ".iqtree" output file that gives the tree's log-likelihood
def get_taxon_likelihood(input_file):
    likelihood = 0
    ll_str = "Log-likelihood of the tree: "
    with open(input_file, "r") as f:
        for line in f.readlines():
            if ll_str in line:
                likelihood = line.split(ll_str)[-1].split(" ")[0]
    return likelihood


# return the distance to the closest leaf of the taxon specified
def calculate_taxon_height(input_tree, taxon_name): 
    tree_cp = input_tree.copy()
    taxon_parent = tree_cp&taxon_name.get_ancestors()[0]
    tree_cp.delete(tree_cp&taxon_name)
    return taxon_parent.get_closest_leaf()[1]


# dictionary to hold the outputs of rules reattach_removed_sequence
sequence_reattachment_data = {}


# Define the workflow
rule all:
    input:
        expand(output_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile", seq_id=get_seq_ids(input_alignment)),
        output_folder+input_alignment+".treefile"


# Define the rule to extract the best model for iqtree on the full MSA
rule model_test_iqtree:
    input:
        msa=input_alignment
    output:
        touch(output_folder+"model-test-iqtree.done"),
        modeltest=output_folder+"input_alignment.fasta_model.iqtree"
    shell:
        """
        iqtree -s {input.msa} --prefix {output_folder}{input.msa}_model -m MF
        """

# Define the rule to extract the model from the IQ-TREE run on the full msa
rule extract_model_for_full_iqtree_run:
    input:
        output_folder+"model-test-iqtree.done",
        msa=rules.model_test_iqtree.output.modeltest
    output:
        model=output_folder+"iqtree-model.txt"
    shell:
        """
        echo $(grep "Best-fit model" {input.msa} | cut -d ":" -f 2) > {output.model}
        """

# Define the rule to remove a sequence from the MSA and write the reduced MSA to a file
rule remove_sequence:
    input:
        msa=input_alignment
    output:
        reduced_msa=temp(output_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta")
    params:
        seq_id=lambda wildcards: wildcards.seq_id
    run:
        # Read the MSA
        with open(input.msa, "r") as f_in:
            sequences = list(SeqIO.parse(f_in, "fasta"))
        
        # Remove the sequence with the specified ID
        reduced_sequences = [seq for seq in sequences if seq.id != params.seq_id]
        
        # Create a directory for the reduced MSA if it doesn't exist
        os.makedirs(os.path.dirname(output.reduced_msa), exist_ok=True)
        
        # Write the reduced MSA to a file
        with open(output.reduced_msa, "w") as f_out:
            SeqIO.write(reduced_sequences, f_out, "fasta")


# Define the rule to run IQ-TREE on the reduced MSA
rule run_iqtree_restricted_alignments:
    input:
        reduced_msa=rules.remove_sequence.output.reduced_msa,
        full_model=rules.extract_model_for_full_iqtree_run.output.model
    output:
        tree=output_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta.treefile"
    shell:
        """
        iqtree -s {input.reduced_msa} -m $(cat {input.full_model}) --prefix {input.reduced_msa}
        """


# Define the rule to run IQ-TREE on the full MSA and get model parameters
rule run_iqtree_on_full_dataset:
    input:
        msa=input_alignment,
        full_model=rules.extract_model_for_full_iqtree_run.output.model
    output:
        tree=output_folder+input_alignment+".treefile"
    shell:
        """
        cp {input.msa} {output_folder}
        iqtree -s {input.msa} -m $(cat {input.full_model}) --prefix {output_folder}{input.msa}
        """


# Define the rule to attach the pruned taxon at each edge
rule reattach_removed_sequence:
    input:
        reduced_tree_nwk=rules.run_iqtree_restricted_alignments.output.tree
    output:
        output_topology=temp(expand(output_folder+"reduced_alignments/{seq_id}/reduced_alignment.fasta_add_at_edge_{edge}.nwk", edge=get_branch_lengths(input.reduced_tree_nwk)))
    params:
        seq_id=lambda wildcards: wildcards.seq_id
    run:
        # open the newick and save the topology
        with open(input.reduced_tree_nwk, "r") as f:
            nh_string = next(f.readlines())
        reduced_topology = Tree(nh_string)

        # for each node(considered as the child of its parent edge), create a copy of
        # the original topology and attach the pruned taxon as a sister of that node
        # (i.e. attach the pruned taxon at that edge), and write new topology to file
        lookup_val=0
        taxon_name="{seq_id}"
        for node in reduced_topology.traverse("postorder"):
            if not node.is_root():
                node.add_features(lookup_key=str(lookup_val)))
                augmented_topology = reduced_topology.copy(method="deepcopy")
                augmented_topology.search_nodes(lookup_key=str(lookup_val)).add_sister(taxon_name)
                augmented_topology.write(format=1, outfile=output.output_topology)
                lookup_val += 1


rule run_iqtree_on_augmented_topologies:
    input:
        msa=input_alignment,
        topology_file=rules.reattach_removed_sequence.output_topology,
        full_model=rules.extract_model_for_full_iqtree_run.output.model
    output:
        alldone=touch(input.topology_file+"_branch_length.done"),
        tree=temp(input.topology_file+"_branch_length.treefile")
    shell:
        """
        iqtree -s {input.msa} -m $(cat {input.full_model}) -te {input.topology_file} --prefix {input.topology_file}_branch_length
        """


# this rule adds a specific key to the global dictionary
rule aggregate_reattachment_data_per_taxon:
    input:
        ready_to_run=rules.run_iqtree_on_augmented_topologies.output.alldone,
        treefile=rules.run_iqtree_on_augmented_topologies.output.treefile,
        mlfile=rules.run_iqtree_on_augmented_topologies.output.iqtree
    output:
        alldone=temp(touch(output_folder+"reduced_aligntments/{seq_id}/aggregate_reattachment_data_per_taxon.done"))
    params:
        seq_id=lambda wildcards: wildcards.seq_id,
        edge=lambda wildcards: wildcards.edge
    run:
       with open(inputfile, "r") as f:
           nh_str = f.readlines()[0]
           this_tree = Tree(nh_str)

       branchlengths = get_branch_lengths(input.treefile)
       taxon_height = calculate_taxon_height(this_tree, {seq_id})
       likelihood = get_taxon_likelihood(input.mlfile)
       topology = nh_str
       
       sequence_reattachment_data["{seq_id}_{edge}"] = [branchlengths,
                                                        taxon_height,
                                                        likelihood,
                                                        topology]


rule write_reattachment_data_to_file:
    inputs:
        expand(output_folder+"reduced_alignments/{seq_id}/aggregate_attachment_data_per_taxon.done", seq_id=get_taxon_)
    outputs:
    run:
        pd.DataFrame(sequence_reattachment_data.items(),\
                     columns=["branch lengths",\
                              "taxon height",\
                              "log-likelihood",\
                              "topology"]\
                     ).to_csv(output_folder+"reduced_alignments/reattachment_data.csv")
