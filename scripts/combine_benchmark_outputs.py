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


summed = df.groupby("rule").sum()
counts = list(df.groupby("rule").count().cpu_time)
rulenames = list(summed.index)

thecol=list(summed.cpu_time)
thetitle="CPU time breakdown"
thename=os.path.join(output_plot_path, "CPU_time_breakdown.pdf")

fig, ax = plt.subplots()
these_bars = ax.bar(rulenames, thecol)
ax.set(title=thetitle)
ax.bar_label(these_bars, labels=["{:d}calls\n{:.3f} per run".format(x, float(list(thecol)[i])/float(x if x != 0 else 1)) for i, x in enumerate(counts)])
plt.xticks(rotation=30, ha="right")
plt.savefig(thename)
