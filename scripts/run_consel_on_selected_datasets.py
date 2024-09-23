import glob
import os
import sys
import numpy as np
import pandas as pd

data_dir = sys.argv[1]
path_to_consel = sys.argv[2]

subdirs = [x.split("/")[-2] + "/" for x in glob.glob(data_dir+"/*/") if os.path.isdir(x + "reduced_alignments/")]

if len(sys.argv) > 3:
    subdir_list = sys.argv[3]
    subdirs = pd.read_csv(subdir_list, header=None)
    #subdirs = [x.split("/")[0] + "/" for x in subdirs[0]]

rerun = False
if len(sys.argv) > 4:
    rerun = True


which_consel = path_to_consel + "/consel"
which_makermt = path_to_consel + "/makermt"

def combine_sitelh_files(f1, f2, output_filename):
    full_str = "Tree	-lnL	Site	-lnL\n"
    with open(f1, "r") as f:
        lines = f.readlines()
        full_str += "\t".join(lines[0].split())+"\n"
        for site, val in enumerate(lines[1].split()[1:]):
            full_str += "\t\t"+str(site + 1) + "\t" + f"{-float(val.strip()):.10f}" + "\n"
    with open(f2, "r") as f:
        lines = f.readlines()
        full_str += "2\t"+str(lines[0].split()[-1])+"\n"
        for site, val in enumerate(lines[1].split()[1:]):
            full_str += "\t\t"+str(site + 1) + "\t" + f"{-float(val.strip()):.10f}" + "\n"
    with open(output_filename, "w") as f:
        f.write(full_str)

for subdir in subdirs[0]:
    dset_name = subdir.split("/")[0]
    taxon_name = subdir.split("/")[-1]
    full_path = data_dir+dset_name
    full_msa = full_path + "full_alignment.fasta"
    full_tree_sitelh = full_msa + ".consel.sitelh"
    pruned_tree = full_msa + ".treefile"

    full_model_file = data_dir + dset_name + "/iqtree-model.txt"
    with open(full_model_file, "r") as f:
        full_model = " ".join(f.readlines())

    taxon_path = full_path + "/reduced_alignments/" + taxon_name + "/"
    reduced_fasta = taxon_path + "without_taxon.fasta"
    reduced_msa = taxon_path + "reduced_alignment.fasta"
    reduced_tree = reduced_msa + ".treefile"
    if not (os.path.isfile(reduced_msa)):
        reduced_msa = reduced_fasta
    inferred_tree_sitelh = reduced_msa + ".consel.sitelh"
    if not (os.path.isfile(inferred_tree_sitelh)):
        inferred_tree_sitelh = taxon_path + "reduced_alignment.fasta.consel.sitelh"

    pruned_tree_base = taxon_path + "pruned_tree.nwk"
    if rerun or (not os.path.isfile(pruned_tree_base + ".consel.sitelh")):
        os.system("head -1 " + taxon_path + "pruned_and_inferred_tree.nwk > " + pruned_tree_base)
        os.system("iqtree -s " + reduced_msa \
                  + " -z " + pruned_tree_base \
                  + " -m " + full_model \
                  + " -n 0 " \
                  + " -wsl --prefix " + pruned_tree_base + ".consel")

    if rerun or (not os.path.isfile(inferred_tree_sitelh)):
        os.system("iqtree -s " + reduced_msa \
                  + " -z " + reduced_tree \
                  + " -m " + full_model \
                  + " -n 0 " \
                  + " -wsl --prefix " + reduced_msa + ".consel")

    combine_sitelh_files(pruned_tree_base + ".consel.sitelh", inferred_tree_sitelh, taxon_path+"consel.txt")
    os.system(which_makermt + " --paup " + taxon_path + "consel.txt")
    os.system(which_consel + " " + taxon_path + "consel")

