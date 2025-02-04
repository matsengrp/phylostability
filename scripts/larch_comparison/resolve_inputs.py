import os
import sys
from ete3 import Tree


newick_file = sys.argv[1]
fasta_file = sys.argv[2]
output_fasta = sys.argv[3]

with open(newick_file, "r") as f:
    tree = Tree(f.readlines()[0])

with open(fasta_file, "r") as f:
    fasta = f.readlines()

leaf_names = [x.name for x in tree.get_leaves()]

with open(output_fasta, "w") as f:
    write_seq=True
    for line in fasta:
        if ">" in line:
            name = line.replace("> ", ">").split(">")[-1].split()[0].strip()
            if name in leaf_names:
                f.write(">" + name + "\n")
                write_seq=True
            else:
                write_seq=False
        else:
            if write_seq:
                f.write(line.strip().replace("-","N") + "\n")
