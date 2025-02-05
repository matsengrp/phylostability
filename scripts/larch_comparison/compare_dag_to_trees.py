from ete3 import Tree
import os
import sys
import historydag as hdag
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

if len(sys.argv) < 6:
    raise RuntimeError("error in compare_dag_to_trees.py: must provide a fasta, a newick file, a protobuf file, a csv, and an output filename")

fastafile = sys.argv[1]
mltreefile = sys.argv[2]
dagfile = sys.argv[3]
csv_filename = sys.argv[4]
output_filename = sys.argv[5]

dataset = mltreefile.split("/")[-3]
cnames = ["dataset", "DAG parsimony score", "tree best parsimony score", "one sided RF", "min RF distance", "max RF distance", "num histories", "average pairwise RF distance", "reference sequence hamming distance"]

df2 = pd.DataFrame(columns=cnames)
if os.path.isfile(csv_filename):
    df2 = pd.read_csv(csv_filename)
if not(dataset in df2["dataset"].values):
    with open(mltreefile, "r") as f:
        nwk = f.readlines()[0]
        ete_tree = Tree(nwk)
        tree = hdag.history_dag_from_newicks([nwk], ["name"])

    def new_fieldname(node, return_label=False):
        if node.is_leaf():
            if return_label:
                return node.label
            return node.label.node_id.split(":")[0]
        else:
            return ""
    cg_dag = hdag.mutation_annotated_dag.load_MAD_protobuf_file(dagfile, compact_genomes=True)
    num_histories = cg_dag.count_histories()
    weight = cg_dag.trim_optimal_weight()
    dag = hdag.HistoryDag(cg_dag.dagroot).add_label_fields(["name"], lambda n: [new_fieldname(n)]).remove_label_fields(["node_id", "compact_genome"])

    allrf = dag.count_sum_rf_distances(tree)
    kwargs = hdag.utils.make_rfdistance_countfuncs(dag, rooted=False, one_sided="right")
    bestrf = tree.optimal_weight_annotate(**kwargs)
    ks = list(allrf.keys())
    vs = list(allrf.values())
    #plt.bar(ks, vs)
    #plt.plot([bestrf, bestrf], [0,max(vs)])
    #plt.title("rf distance breakdown for all trees in the DAG to ML tree")
    #plt.xlabel("rf distance")
    #plt.ylabel("count")
    #plt.tight_layout()
    #plt.savefig(output_filename)
    #plt.clf()

    fasta = {}
    seq_len=0
    with open(fastafile, "r") as f:
        current_seq=""
        for line in f:
            if ">" in line:
                if len(current_seq) > 0:
                    if seq_len < 1:
                        seq_len=len(current_seq)
                    fasta[current_key] = current_seq[:seq_len]
                    current_seq=""
                current_key = line.replace("> ", ">").split(">")[-1].split()[0]
            else:
                current_seq = current_seq + line.strip().replace("-","N")
        fasta[current_key] = current_seq[:seq_len]
    
    for l in ete_tree.traverse():
        if l.is_leaf():
            l.add_features(sequence=fasta[l.name])
        else:
            l.add_features(sequence="N"*seq_len)

    disamb_tree = hdag.parsimony.disambiguate(ete_tree)
    tree_best_parsimony_score = hdag.parsimony.parsimony_score(disamb_tree)

    r1 = cg_dag.get_reference_sequence()
    r2 = disamb_tree.get_tree_root().sequence
    refseq_differences = sum(1 for i, x in enumerate(r1) if x != r2[i])

    vals = [[dataset], [weight], [tree_best_parsimony_score], [bestrf],  [min(ks)], [max(ks)], [num_histories], [dag.average_pairwise_rf_distance()], [refseq_differences]]

    df1 = pd.DataFrame.from_dict(dict(zip(cnames, vals)))
    df = pd.concat([df2, df1], ignore_index=True)
    df.to_csv(csv_filename, index=False)
