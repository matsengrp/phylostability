import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

from sklearn.metrics import confusion_matrix, auc, ConfusionMatrixDisplay

# Font sizes for figures
plt.rcParams.update({"font.size": 12})  # Adjust this value as needed
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16

# Colour for plots
dark2 = mpl.colormaps["Dark2"]


def plot_au_test_pie_chart(df, plot_filepath):
    def custom_format(values):
        min_value = min(values)
        new_values = []
        for value in values:
            if value == 0:
                return "0%"
            else:
                # Determine the number of decimal places needed
                decimal_places = abs(int(math.floor(math.log10(abs(value)))))
                if decimal_places == 0:
                    decimal_places += 1
                format_string = "{{:.{}f}}%".format(decimal_places)
                new_values.append(format_string.format(value))
        return new_values

    # Pie Chart
    condition1 = ((df["normalised_tii"] == 0.0) & (df["p-AU"] < 0.05)).sum()
    condition2 = ((df["normalised_tii"] > 0.0) & (df["p-AU"] < 0.05)).sum()
    condition3 = ((df["normalised_tii"] == 0.0) & (df["p-AU"] >= 0.05)).sum()
    condition4 = ((df["normalised_tii"] > 0.0) & (df["p-AU"] >= 0.05)).sum()

    # Data to plot
    sizes = [condition1, condition2, condition3, condition4]
    total = sum(sizes)
    percentages = custom_format([100 * (size / total) for size in sizes])

    labels = [
        "stable and significant ({}) \n".format(percentages[0]),
        "unstable and significant ({})".format(percentages[1]),
        "stable and non-significant ({})".format(percentages[2]),
        "unstable and non-significant ({})".format(percentages[3]),
    ]
    # Colors
    colors = [dark2.colors[i] for i in range(3, -1, -1)]
    # Exploding the 1st slice (optional)
    explode = (0.1, 0, 0, 0)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        # autopct="%1.1f%%",
        # shadow=True,
        startangle=140,
    )
    plt.axis("equal")
    plt.savefig(plot_filepath)


all_au_test_results = snakemake.input.all_au_test_results
pie_plot_file = snakemake.output.pie_plot_file

df = pd.read_csv(all_au_test_results)
plot_au_test_pie_chart(df, pie_plot_file)
