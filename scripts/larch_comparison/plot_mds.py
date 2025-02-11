import sys
import numpy as np
import matplotlib.pyplot as plt

def plot_mds(mds_file, dim, subsets = [], subset_names = [], plot_filename="", color_vec=[]):
    mds_data = np.memmap(mds_file, dtype='float32', mode='r', shape=dim)
    if len(subsets) < 1:
        plt.clf()
        if len(color_vec) > 0:
            plt.scatter(mds_data[:,0], mds_data[:,1], c=color_vec)
            plt.colorbar()
        else:
            plt.scatter(mds_data[:,0], mds_data[:,1])
        plt.savefig(plot_filename, format="svg")
        plt.savefig(plot_filename.split(".svg")[0]+".png")
    else:
        plt.clf()
        marks="+oxv^"*len(subset_names)

        # make sure the dimensions of the scatterplot encompass all the data
        plt.scatter(mds_data[:,0], mds_data[:,1], label=None, alpha=0.01)

        for i, sn in enumerate(subset_names):
            if len(color_vec) > 0:
                plt.scatter(mds_data[subsets[i],0], mds_data[subsets[i],1], label=sn, marker=marks[i], c=[color_vec[j] for j in subsets[i]], alpha=0.6)
            else:
                plt.scatter(mds_data[subsets[i],0], mds_data[subsets[i],1], label=sn, marker=marks[i])
        if len(color_vec) > 0:
            plt.colorbar()
        plt.legend()
        plt.savefig(plot_filename, format="svg")
        plt.savefig(plot_filename.split(".svg")[0]+".png")

mds_file = sys.argv[1] # name of path to mds file
dim_file = sys.argv[2] # dimensions of mds file
subset_1_file = sys.argv[3] # name of file that contains the number of trees in the first subset
output_filename = sys.argv[4] # name of filepath to the output image to
colors = []
if len(sys.argv) > 5:
    color_filename = sys.argv[5] # name of filepath with labels for the colors of each tree
    with open(color_filename, "r") as f:
        colors = [int(x.strip()) for x in f.readlines()]
    mcol = max(colors)
    colors.append(mcol)
    colors.append(mcol)
    colors.append(mcol)

if output_filename.endswith(".png") or output_filename.endswith(".csv"):
    output_filename = output_filename[:-4] + ".svg"

with open(dim_file, "r") as f:
    dim = tuple([int(x) for x in f.readlines()[0].split()])

with open(subset_1_file, "r") as f:
    subset_1_trees = list(range(int(f.readlines()[0].split()[0])))
other_trees = list(range(len(subset_1_trees), dim[0]))

if len(colors) > 0:
    plot_mds(mds_file, (dim[0], 2), subsets=[subset_1_trees, other_trees], subset_names = ["larch trees", "ML tree"], plot_filename=output_filename, color_vec=colors)
else:
    plot_mds(mds_file, (dim[0], 2), subsets=[subset_1_trees, other_trees], subset_names = ["larch trees", "ML tree"], plot_filename=output_filename)
