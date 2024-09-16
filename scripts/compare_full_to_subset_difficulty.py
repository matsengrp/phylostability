import os
import glob
import sys
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

num_bins = int(sys.argv[1])
df_name = sys.argv[2]
data_path = sys.argv[3]
raxml_path = sys.argv[4]
fasta_name = sys.argv[5]
num_sets_to_sample_per_range = int(sys.argv[6])

bins = np.linspace(0, 1, num_bins+1)
difficulties = []

df = pd.read_csv(df_name, index_col=0)

for i in range(num_bins):
    loval = bins[i]
    hival = bins[i+1]
    dsets_in_difficulty_range = df.loc[(df["difficulty"] >= loval) & (df["difficulty"] < hival), "name"].tolist()
    chosen_dsets = random.choices(dsets_in_difficulty_range, k = num_sets_to_sample_per_range)
    for dset in chosen_dsets:
        full_path = data_path+"/"+dset+"/reduced_alignments/"
        if not all(os.path.isfile(x+"/pythia_difficulty.txt") for x in glob.glob(full_path+"/*/")):
            os.system("./calculate_difficulties.sh " \
                      + full_path \
                      + " " + fasta_name \
                      + " " + raxml_path)
        for removed_taxon in glob.glob(full_path+"/*/"):
            with open(removed_taxon + "/pythia_difficulty.txt", "r") as f:
                difficulty = float(f.readlines()[0].strip())
                difficulties.append([difficulty, "reduced MSA", i])

        with open(data_path+"/"+dset+"/pythia_difficulty.txt", "r") as f:
            difficulty = float(f.readlines()[0].strip())
            difficulties.append([difficulty, "full MSA", i])

full_difficulties = pd.DataFrame(difficulties, columns=["difficulty", "MSA type", "bin"])
sns.violinplot(full_difficulties, x="bin", y="difficulty", hue="MSA type")
plt.savefig("subsets_difficulties_violinplot.png")
plt.clf()

sns.boxplot(full_difficulties, x="bin", y="difficulty", hue="MSA type")
plt.savefig("subsets_difficulties_boxplot.png")
plt.clf()

