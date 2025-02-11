import historydag as hdag
import sys
import random

dagfile = sys.argv[1]
num_histories_to_extract = int(sys.argv[2])
outputfile = sys.argv[3]
prefix_for_newick = sys.argv[4]

cg_dag = hdag.mutation_annotated_dag.load_MAD_protobuf_file(dagfile, compact_genomes=True)
num_histories = cg_dag.count_histories()
weight = cg_dag.trim_optimal_weight()

def new_fieldname(node, return_label=False):
    if node.is_leaf():
        if return_label:
            return node.label
        return node.label.node_id.split(":")[0]
    else:
        return ""
dag = hdag.HistoryDag(cg_dag.dagroot).add_label_fields(["name"], lambda n: [new_fieldname(n)])
ua_child_muts = [(len(p.label.compact_genome.mutations),p) for p in dag.dagroot.children()]
max_val = max(next(zip(*ua_child_muts)))
ua_MP_children = [x[1] for x in ua_child_muts if x[0] == max_val]

if num_histories_to_extract > dag.count_histories():
    print("Warning in extract_newicks_subset.py: not enough histories in DAG to have unique sample\n")


#node_num = {n:"s"+str(i) for i, n in enumerate(list(dag.postorder()))}
#def numbered_fieldname(node):
#    if node.is_leaf():
#        return node.label.name
#    else:
#        return node_num[node]

with open(outputfile, "w") as f:
    for i in range(num_histories_to_extract):
        n = random.sample(ua_MP_children, 1)
        #tree_str = prefix_for_newick + " " + dag.sample_with_node(n[0]).to_newick(lambda n: numbered_fieldname(n), features=[])
        tree_str = prefix_for_newick + " " + dag.sample_with_node(n[0]).to_newick(lambda n: new_fieldname(n), features=[])
        f.write(tree_str + "\n")
