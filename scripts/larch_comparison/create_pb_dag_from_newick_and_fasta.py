import historydag as hdag
import sys

fasta_file = sys.argv[1]
newick_file = sys.argv[2] # e.g. uncondensed-final-tree.nh
output_filename = sys.argv[3]

with open(newick_file, "r") as f:
    newick = f.readlines()[0]

fasta = {}
root_name = ""
with open(fasta_file, "r") as f:
    key = ""
    val = ""
    for line in f.readlines():
        if ">" in line:
            if len(val) > 0:
                fasta[key] = val.replace("-","N")
                val = ""
            key = line.split(">")[-1].strip()
            if len(root_name) < 1:
                root_name = key
        else:
            val = val + line.strip()
    fasta[key] = val.replace("-","N")
def node_names(n):
    if n.is_leaf():
        return n.label.name
    else:
        return ""

tree = hdag.parsimony.build_tree(newick, fasta)

disamb_tree = hdag.parsimony.disambiguate(tree)
disamb_dag = hdag.history_dag_from_etes([disamb_tree], ["sequence", "name"])
cg_dag = hdag.mutation_annotated_dag.CGHistoryDag.from_history_dag(disamb_dag, reference=fasta[root_name]).add_label_fields(["node_id"], lambda n: [node_names(n)])
cg_dag.to_protobuf_file(output_filename)

