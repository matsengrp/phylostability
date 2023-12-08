import pandas as pd
import matplotlib.pyplot as plt


iqtree_files=snakemake.input.iqtree_file
plots_folder=snakemake.params.plots_folder

def extract_table_from_file(filename):
    # Flag to indicate whether we are currently reading the table
    reading_table = False
    table_data = []

    with open(filename, 'r') as file:
        for line in file:
            # Check for the start of the table
            if line.startswith("Tree      logL"):
                reading_table = True
                continue  # Skip the header line

            # Check for the end of the table
            if line.startswith("deltaL"):
                break
            
            if line.startswith("------"):
                continue

            # Read table data
            if reading_table and line.strip():
                this_line = [l for l in line.split() if l not in ["+", "-"]]
                table_data.append(this_line)
    # Convert the table data to a pandas DataFrame
    columns = ["Tree", "logL", "deltaL", "bp-RELL", "p-KH", "p-SH", "c-ELW"]
    df = pd.DataFrame(table_data, columns=columns)
    print(df)

    # Convert numerical columns from string to appropriate types
    for col in ["logL", "deltaL", "bp-RELL", "p-KH", "p-SH", "c-ELW"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

df_list = []
for iqtree_file in iqtree_files:
    df = extract_table_from_file(iqtree_file)
    print(df)
    df["filename"] = iqtree_file.split("/")[1]
    df_list.append(df)

print(df_list)
big_df = pd.concat(df_list, ignore_index=True)

print(big_df)

# Group by 'filename' and calculate the proportion
grouped_df = big_df.groupby('filename', group_keys=True)
proportion_df = grouped_df.apply(lambda x: (x['p-SH'] < 0.05).sum() / len(x)).reset_index(name='proportion')

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(proportion_df['filename'], proportion_df['proportion'])
plt.xlabel('Filename')
plt.ylabel('Proportion of p-SH < 0.05')
plt.xticks(rotation=45, ha='right')  # Rotate the x-axis labels for better readability
plt.title('Proportion of p-SH < 0.05 for each file')
plt.tight_layout()  # Adjust layout for better fit
plt.savefig(plots_folder+'p_sh_proportion_plot.pdf')
