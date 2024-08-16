import sys
import numpy as np

f1 = snakemake.input.full_tree_sitelh
f2 = snakemake.input.inferred_tree_sitelh
output_filename = snakemake.output.both_trees_txtfile

full_str = "Tree	-lnL	Site	-lnL\n"
with open(f1, "r") as f:
    lines = f.readlines()
    full_str += "\t".join(lines[0].split())+"\n"
    for site, val in enumerate(lines[1].split()[1:]):
        full_str += "\t\t"+str(site + 1) + "\t" + str(-float(val.strip())) + "\n"

with open(f2, "r") as f:
    lines = f.readlines()
    full_str += "2\t"+str(lines[0].split()[-1])+"\n"
    for site, val in enumerate(lines[1].split()[1:]):
        full_str += "\t\t"+str(site + 1) + "\t" + str(-float(val.strip())) + "\n"

with open(output_filename, "w") as f:
    f.write(full_str)
