import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import glob
import os

plt.rcParams.update({"font.size": 12})  # Adjust this value as needed
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

# Colour for plots
dark2 = mpl.colormaps["Dark2"]

main_dir = sys.argv[1]
plot_path = sys.argv[2]
csv_path = sys.argv[3]
msa_name = []
msa_difficulty = []
for subdir in glob.glob(main_dir+"/*/"):
    if os.path.isfile(subdir+"pythia_difficulty.txt"):
        msa_name.append(subdir.split("/")[-2])
        with open(subdir+"pythia_difficulty.txt", "r") as f:
            msa_difficulty.append(float(f.readlines()[0].strip()))
msa_difficulties = pd.DataFrame({"name":msa_name, "difficulty":msa_difficulty})
msa_difficulties.to_csv(csv_path)

bin_width = 0.02

# Calculate bin edges, offset to center bins around integers
min_bin = min(msa_difficulty) - (min(msa_difficulty) % bin_width) - (bin_width / 2)
max_bin = max(msa_difficulty) + (bin_width - (max(msa_difficulty) % bin_width)) + (bin_width / 2)
bins = np.arange(min_bin, max_bin, bin_width)

plt.figure()
sns.histplot(data=msa_difficulties, x="difficulty", color=dark2.colors[0], bins=bins)
plt.xlabel("difficulty score")
# plt.title("difficulty breakdown for selected MSA sets")
plt.tight_layout()
plt.savefig(plot_path)
plt.clf()
