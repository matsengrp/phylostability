import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

main_dir = sys.argv[1]
plot_path = sys.argv[2]
csv_path = sys.argv[3]
msa_name = []
msa_difficulty = []
for subdir in glob.glob(main_dir+"/*/"):
    if os.path.isfile(subdir+"pythia_difficulty.txt"):
        msa_name.append(subdir.split("/")[-1])
        with open(subdir+"pythia_difficulty.txt", "r") as f:
            msa_difficulty.append(float(f.readlines()[0].strip()))
msa_difficulties = pd.DataFrame({"name":msa_name, "difficulty":msa_difficulty})
msa_difficulties.to_csv(csv_path)

plt.figure()
sns.histplot(data=msa_difficulties, x="difficulty")
plt.title("difficulty breakdown for selected MSA sets")
plt.savefig(plot_path)
plt.clf()
