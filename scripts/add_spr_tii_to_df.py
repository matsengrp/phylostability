import pandas as pd
from ete3 import Tree
import subprocess
import os


input_csv = snakemake.input.csv
spr_files = snakemake.input.spr_files
output_csv = snakemake.output.csv


csv = pd.read_csv(input_csv, index_col=0)
print(csv)
taxon_list = csv.index.to_list()

spr_tiis = []
for taxon in taxon_list:
    spr_file = [f for f in spr_files if "/" + taxon + "/" in f][0]

    spr_str = "d_USPR ="
    with open(spr_file, "r") as f:
        for line in reversed(f.readlines()):
            if spr_str in line.strip():
                spr_dist = line.strip().split("=")[1]
                break
    spr_tiis.append(spr_dist)
csv["tii"] = spr_tiis
print(csv)
csv.to_csv(output_csv)
