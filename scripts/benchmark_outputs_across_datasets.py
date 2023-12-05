import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df_names = snakemake.input.dfs
fasta_names = snakemake.input.fastas
plots_folder = snakemake.params.plots_folder


def extract_rule_names_from_df(df_name):
    df = pd.read_csv(df_name)
    return df.rule.unique().tolist()
rule_names = extract_rule_names_from_df(df_names[0])


values_to_plot = {"dataset": [],
                  "# sequences": [],
                  "sequence length": [],
                  "CPU time": [],
                  "time in seconds" : []}
for dfn in df_names:
    df = pd.read_csv(dfn).fillna(0)
    dset = dfn.split("/")[1]
    fastan = [f for f in fasta_names if dset in f][0]
    with open(fastan, "r") as f:
        ctr = 0
        seq_len = 0
        for line in f:
            if ">" in line:
                ctr += 1
            elif ctr < 2:
                seq_len += len(line)
        num_seqs = ctr

    values_to_plot["dataset"].append(dset)
    values_to_plot["# sequences"].append(num_seqs)
    values_to_plot["sequence length"].append(seq_len)
    values_to_plot["CPU time"].append(sum(df["cpu_time"]))
    values_to_plot["time in seconds"].append(sum(df["s"]))
df = pd.DataFrame(values_to_plot)
mx = df["# sequncees"].max() if df["# sequences"].max() != 0 else 1
mn = df["# sequncees"].min()
df["normalized # sequences"] = [20*2**((x-mn)/mx) for x in df["# sequences"]]

def plot_total_value_breakdown(df, colname, plot_path):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df.sort_values(colname), x="dataset", y=colname, hue="sequence length", size="# sequences")
    plt.xticks(
        rotation=90,
    )
    plt.title(colname + " breakdown over all datasets")
    plt.xlabel("data set")
    plt.ylabel(colname)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.clf()

plot_filepath = os.path.join(plots_folder, "CPU_time_breakdown_across_all_datasets.pdf")
plot_total_value_breakdown(df, "CPU time", plot_filepath)

plot_filepath = os.path.join(plots_folder, "runtime_breakdown_across_all_datasets.pdf")
plot_total_value_breakdown(df, "time in seconds", plot_filepath)
