import os
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt

output_file = snakemake.output.output_file
output_plot_path = snakemake.params.output_plot_path
files_to_join = [ \
  snakemake.params.iqtree_model, \
  snakemake.params.remove_taxon, \
  snakemake.params.iqtree_whole_taxon_set, \
  snakemake.params.iqtree_restricted_taxon_set, \
  snakemake.params.reattach_taxon, \
  snakemake.params.iqtree_augmented_topologies \
]

def string_intersect(a, b):
    def _iter():
        for aa, bb in zip(a, b):
            if aa == bb:
                yield aa
            else:
                 return
    return "".join(_iter())

dfs = []
for filename in files_to_join:
    if isinstance(filename, str):
        df = pd.read_csv(filename, sep='\t')
        df["rule"] = filename.split("/benchmark_")[-1].split(".txt")[0]
    else:
        matching_filename = "".join(reduce(string_intersect, filename))
        df = pd.concat((pd.read_csv(ff, sep='\t') for ff in filename), ignore_index=True)
        df["rule"] = matching_filename.split("/benchmark_")[-1]
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
df.to_csv(output_file)


df["count"] = 1

colname = "cpu_time"
summed = df.groupby("rule").sum().sort_values(colname)
counts = list(summed["count"])
rulenames = list(summed.index)
thename=os.path.join(output_plot_path, "CPU_time_breakdown.pdf")
thecol=list(summed[colname])
thetitle="CPU time breakdown\n total time: {}".format(sum(thecol))
fig, ax = plt.subplots(figsize=(10,5))
these_bars = ax.bar(rulenames, thecol, width=0.8)
ax.set(title=thetitle)
ax.bar_label(these_bars, labels=["{:d}calls\n{:.3f} per run".format(x, float(list(thecol)[i])/float(x if x != 0 else 1)) for i, x in enumerate(counts)])
plt.xticks(rotation=30, ha="right")
plt.ylabel(colname)
plt.ylim(min(thecol) - 0.5, max(thecol)*1.2)
fig.tight_layout()
plt.savefig(thename)
