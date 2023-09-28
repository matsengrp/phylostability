import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


taxon_df_csv = snakemake.input.taxon_df_csv
taxon_edge_df_csv = snakemake.input.taxon_edge_df_csv

taxon_df = pd.read_csv(taxon_df_csv)

sns.scatterplot(data = taxon_df, x = "edpl", y = "tii")
plt.savefig("plots/edpl_vs_tii.pdf")
plt.clf()

for csv in taxon_edge_df_csv:
    seq_id = csv.split("/")[-2]
    taxon_edge_df = pd.read_csv(csv)
    
    # swarmplot of likelihood ratios
    sns.swarmplot(data = taxon_edge_df["likelihood_ratio"])
    plt.savefig("plots/" + seq_id + "/likelihood_ratio_swarmplot_seq.pdf")
    plt.clf()



# Create output plot
