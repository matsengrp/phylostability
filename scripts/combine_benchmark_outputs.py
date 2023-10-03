import pandas as pd
from functools import reduce

output_file = snakemake.output.output_file
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
        df["file"] = filename.split("/")[-1]
    else:
        matching_filename = "".join(reduce(string_intersect, filename))
        df = pd.concat((pd.read_csv(ff, sep='\t') for ff in filename), ignore_index=True)
        df["file"] = matching_filename.split("/")[-1]
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
df.to_csv(output_file)
