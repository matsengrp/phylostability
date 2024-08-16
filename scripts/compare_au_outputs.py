import os
import sys
import glob
import math
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

the_dir = sys.argv[1]
full_pv_file = sys.argv[2] # e.g. au_test_result.csv
path_to_consel = sys.argv[3] # e.g. consel/bin/

subdirs = [x for x in glob.glob(the_dir+"/*/") if os.path.isfile(x + "full_alignment.fasta.consel.sitelh")]
if os.path.isfile(full_pv_file.split(".csv")[0]+"_with_consel_pv.csv"):
    full_pv_df = pd.read_csv(full_pv_file.split(".csv")[0] + "_with_consel_pv.csv", index_col=0)
else:
    full_pv_df = pd.read_csv(full_pv_file, index_col=0)
full_pv_df["consel-p-AU"] = -1.0

def read_pv_from_file(fn, col_to_read=3):
    print(fn)
    with open(fn, "r") as f: 
        return f.readlines()[3].split()[col_to_read].strip()

for subdir in subdirs:
    ds_name = subdir.split("/")[-2]
    #if all(os.path.isfile(t_p + "consel.pv") for t_p in glob.glob(subdir+"/reduced_alignments/*/")):
    for taxon_path in glob.glob(subdir+"/reduced_alignments/*/"):
        taxon_name = taxon_path.split("/")[-2]
        if os.path.isfile(taxon_path + "consel.pv"):
            if not (os.path.isfile(taxon_path + "consel_pv_output")):
                consel_output = taxon_path + "consel.pv"
                os.system(path_to_consel + "catpv " \
                          + consel_output \
                          + " | sed -r 's/#|//g' > " + taxon_path + "consel_pv_output")

            pv_output = float(read_pv_from_file(taxon_path+"consel_pv_output"))
            full_pv_df.loc[(full_pv_df["dataset"] == ds_name) & (full_pv_df["seq_id"] == taxon_name), "consel-p-AU"] = pv_output

full_pv_df.to_csv(full_pv_file.split(".csv")[0] + "_with_consel_pv.csv")

def custom_format(values):
    min_value = min(values)
    new_values = []
    for value in values:
        if value == 0:
            new_values.append("0%")
        else:
            decimal_places = abs(int(math.floor(math.log10(abs(value))))) 
            if decimal_places < 2:
                decimal_places = 2
            format_string = "{{:.{}f}}%".format(decimal_places)
            new_values.append(format_string.format(value))
    return new_values

df = full_pv_df.loc[full_pv_df["consel-p-AU"] >= 0]

condition1 = ((df["normalised_tii"] == 0.0).sum())
condition2 = ((df["normalised_tii"] > 0.0) & (df["consel-p-AU"] < 0.05)).sum()
condition3 = ((df["normalised_tii"] > 0.0) & (df["consel-p-AU"] >= 0.05)).sum()
sizes = [condition1, condition2, condition3]
total = sum(sizes)
percentages = custom_format([100 * (size / total) for size in sizes])
labels = [
    "stable ({}) \n".format(percentages[0]),
    "unstable & significant ({}) \n".format(percentages[1]),
    "unstable & non-significant ({}) \n".format(percentages[2]),
]
dark2 = mpl.colormaps["Dark2"]
colors = [dark2.colors[i] for i in [2,1,0]]
plt.figure(figsize=(12, 4))
plt.pie(sizes, labels=labels, colors=colors, startangle=140)
plt.axis("equal")
plt.savefig("consel-p-AU-pie.png")

plt.clf()

full_pv_df.loc[:, ["p-AU", "consel-p-AU"]].plot.hist(alpha=0.5)
plt.savefig("p-value-comparisons.png")

plt.clf()

full_pv_df.loc[full_pv_df["consel-p-AU"] >= 0, ["p-AU", "consel-p-AU"]].plot.hist(alpha=0.5)
plt.savefig("nonmissing-p-value-comparisons.png")
