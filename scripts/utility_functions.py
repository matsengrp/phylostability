from Bio import SeqIO
from ete3 import Tree

def write_restricted_fastas(input_msa, seq_id):
    # Read the MSA
    with open(input_msa, "r") as f_in:
        sequences = list(SeqIO.parse(f_in, "fasta"))
    
    # Remove the sequence with the specified ID
    reduced_sequences = [seq for seq in sequences if seq.id != seq_id]
    
    # Create a directory for the reduced MSA if it doesn't exist
    os.makedirs(os.path.dirname(output.reduced_msa), exist_ok=True)
    
    # Write the reduced MSA to a file
    with open(output.reduced_msa, "w") as f_out:
        SeqIO.write(reduced_sequences, f_out, "fasta")


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
                break
    return likelihood


# return the distance to the closest leaf of the taxon specified
def calculate_taxon_height(input_tree, taxon_name): 
    tree_cp = input_tree.copy()
    taxon_parent = tree_cp&taxon_name.get_ancestors()[0]
    tree_cp.delete(tree_cp&taxon_name)
    return taxon_parent.get_closest_leaf()[1]


def write_reduced_fastas(input_msa, output_msa, seq_id):
    # Read the MSA
    with open(input_msa, "r") as f_in:
        sequences = list(SeqIO.parse(f_in, "fasta"))
    
    # Remove the sequence with the specified ID
    reduced_sequences = [seq for seq in sequences if seq.id != seq_id]
    
    # Create a directory for the reduced MSA if it doesn't exist
    os.makedirs(os.path.dirname(output_msa), exist_ok=True)
    
    # Write the reduced MSA to a file
    with open(output_msa, "w") as f_out:
        SeqIO.write(reduced_sequences, f_out, "fasta")
