import sys
from ete3 import Tree
import numpy as np
import historydag as hdag

dm_file=sys.argv[1]
tree1_idx=sys.argv[2]
tree2_idx=sys.argv[3]
tree1_str=sys.argv[4]
tree2_str=sys.argv[5]
file1_num_trees=sys.argv[6]
file2_num_trees=sys.argv[7]

data_mat = np.memmap(dm_file, dtype='float32', mode='r+', shape=(int(file1_num_trees), int(file2_num_trees)))
def get_distance(nw1, nw2):
    t1 = hdag.history_dag_from_newicks([nw1], label_features=["name"])
    t2 = hdag.history_dag_from_newicks([nw2], label_features=["name"])
    return t1.sum_rf_distances(t2)

data_mat[int(tree1_idx),int(tree2_idx)] = get_distance(tree1_str, tree2_str)
#data_mat.flush()
