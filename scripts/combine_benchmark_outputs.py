from Bio import SeqIO
import os
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt

output_file = snakemake.output.output_file
output_plot_path = snakemake.params.output_plot_path
benchmarking_folder = snakemake.params.benchmarking_folder
subdir = snakemake.params.subdir
files_to_join = [ \
  x for x in os.listdir(benchmarking_folder)
]
print(subdir + "/full_alignment.fasta")
seq_ids = [record.id for record in SeqIO.parse(subdir + "/full_alignment.fasta", "fasta")]


files_met = []
dfs = []
for filename in files_to_join:
    rulename = filename.split("/benchmark_")[-1].split(".txt")[0]
    rule_for_sequence = False
    for sid in seq_ids:
        if sid in rulename:
            rulename = rulename.split("_"+sid)[0]
            rule_for_sequence = True
            break
    if rulename not in files_met:
        if rule_for_sequence:
            df = pd.concat((pd.read_csv(benchmarking_folder+ff, sep='\t') for ff in files_to_join if rulename in ff), ignore_index=True)
        else:
            df = pd.read_csv(benchmarking_folder+filename, sep='\t')
        df["rule"] = rulename
        files_met.append(rulename)
        dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
df.to_csv(output_file)


df["count"] = 1

def bar_plot_breakdown(dataframe, colname, filename):
    summed = dataframe.groupby("rule").sum().sort_values(colname)
    counts = list(summed["count"])
    rulenames = list(summed.index)
    thename=os.path.join(output_plot_path, filename)
    thecol=list(summed[colname])
    thetitle="{} breakdown\n total time: {}".format(colname, sum(thecol))

    fig, ax = plt.subplots(figsize=(10,5))
    these_bars = ax.bar(rulenames, thecol, width=0.8)
    ax.set(title=thetitle)
    ax.bar_label(these_bars, labels=["{:d}calls\n{:.3f} per run".format(x, float(list(thecol)[i])/float(x if x != 0 else 1)) for i, x in enumerate(counts)])
    plt.xticks(rotation=90, ha="right")
    plt.ylabel(colname)
    plt.ylim(min(thecol) - 0.5, max(thecol)*1.2)
    fig.tight_layout()
    plt.savefig(thename)
    return sum(thecol)

tot_cpu_time = bar_plot_breakdown(df, "cpu_time", "CPU_time_breakdown.pdf")
tot_runtime = bar_plot_breakdown(df, "s", "runtime_breakdown.pdf")
