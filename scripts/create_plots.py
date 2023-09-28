import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


taxon_df_csv = snakemake.input.taxon_df_csv
taxon_edge_df_csv = snakemake.input.taxon_edge_df_csv

taxon_df = pd.read_csv(taxon_df_csv, index_col=0)
taxon_df.index.name ="taxon_name"
print(taxon_df)

sns.scatterplot(data = taxon_df, x = "edpl", y = "tii")
plt.savefig("plots/edpl_vs_tii.pdf")
plt.clf()

# sort DF by TII values
taxon_df = taxon_df.sort_values(by='tii')
print(taxon_df)

# for taxon_name in sorted_df[]
for csv in taxon_edge_df_csv:
    seq_id = csv.split("/")[-2]
    taxon_edge_df = pd.read_csv(csv)
    
    # swarmplot of likelihood ratios
    sns.swarmplot(data = taxon_edge_df["likelihood"])
    plt.savefig("plots/" + seq_id + "/likelihood_swarmplot_seq.pdf")
    plt.clf()



# Create output plot
