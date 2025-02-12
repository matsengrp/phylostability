import sys
import historydag as hdag
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

point_data_filename=sys.argv[1]
point_data_dim=sys.argv[2]
cluster_label_filename=sys.argv[3]
cluster_centroid_filename=sys.argv[4]
cluster_centroid_indices_filename=sys.argv[5]
newicks_filename=sys.argv[6]

best_k = 1
if len(sys.argv) > 7:
    with open(sys.argv[7], "r") as f:
        best_k=int(f.readlines()[0].strip())


# get dimension of the data
dim=(int(point_data_dim), 2)

# load the MDS data
point_mat = np.memmap(point_data_filename, dtype='float32', mode='r+', shape=dim)
points = np.array([[y for y in x] for x in point_mat[:,:]])

def make_frac(a, b):
    if b != 0:
        return float(a)/b
    elif abs(a) > tol:
        return np.inf
    else:
        return 0.0
def get_best_k(data, k_max=12):
    best_k = 2
    sek_vals = [] # silhouette score, elbow-value, and k, used to find best cluster
    for k in range(2, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        labels = kmeans.predict(data)
        sek_vals.append((silhouette_score(data, labels), kmeans.inertia_, k))

    best_s_locs = [x[-1] for x in  sorted(sek_vals)[-2:]]
    slope_ratios = [(make_frac(sek_vals[i][1] - x[1],x[1] - sek_vals[i+2][1]), i + 1) for i, x in enumerate(sek_vals[1:-1])]
    best_e_locs = [x[-1] + 2 for x in sorted(slope_ratios)[-2:]]

    # if inertia is overall significantly low, or if the overally curve is near-linear, there is no elbow
    biggest_slope_change = slope_ratios[best_e_locs[-1] - 3][0]
    if max([x[1] for x in sek_vals]) < 0.005 or biggest_slope_change < .5:
        best_e_locs = [2]
    fig,ax1=plt.subplots()

    ax1.set_xlabel("k")
    ax1.set_ylabel("silhouette score", color='tab:blue')
    ax1.scatter([x[-1] for x in sek_vals], [x[0] for x in sek_vals], color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.scatter([sek_vals[x - 2][-1] for x in best_s_locs], [sek_vals[x - 2][0] for x in best_s_locs], color='tab:red', marker='o')
    ax2 = ax1.twinx()
    ax2.set_ylabel("elbow score", color='tab:gray')
    ax2.plot([x[-1] for x in sek_vals], [x[1] for x in sek_vals], color='tab:gray')
    ax2.scatter([sek_vals[x-2][-1] for x in best_e_locs], [sek_vals[x-2][1] for x in best_e_locs], color='tab:gray', marker='x')
    ax2.tick_params(axis='y', labelcolor='tab:gray')
    fig.tight_layout()
    plt.savefig(cluster_label_filename + "_scores.png")

    intersection = set(best_e_locs).intersection(set(best_s_locs))
    return best_s_locs[-1] if best_s_locs[-1] in intersection else best_s_locs[-2]

# identify the clusters in the larch MDS plot
if best_k == 1:
    best_k = get_best_k(points[:-1])
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(points[:-1])
labels = kmeans.predict(points[:-1])

# write labels to file
with open(cluster_label_filename, "w") as f:
    for i, l in enumerate(labels):
        f.write(str(i) + ", " + str(l) + "\n")

# for each cluster, find a 'centroid' tree
centroid_topologies = []
def find_closest_point(point, data):
    ds = np.linalg.norm(data - np.array(point), axis=1)
    # return index in ds of the smallest value
    return sorted(list(zip(ds, range(len(ds)))))[0][-1]

for centroid in kmeans.cluster_centers_:
    closest_tree_index = find_closest_point(centroid, points[:-1])
    centroid_topologies.append(closest_tree_index)

nwks = []
with open(newicks_filename, "r") as f:
    nwks = f.readlines()

# write tree topology for each cluster's centroid to a newick file
with open(cluster_centroid_filename, "w") as f:
    for centroid_topology in centroid_topologies:
        first_paren = nwks[centroid_topology].index('(')
        f.write(nwks[centroid_topology][first_paren:] + "\n")

# write index of tree for each cluster's centroid to a newick file
with open(cluster_centroid_indices_filename, "w") as f:
    for centroid_topology in centroid_topologies:
        f.write(str(centroid_topology) + "\n")

# write labels to but with centroids given their own label
with open(cluster_label_filename+"_centroid_annotated", "w") as f:
    max_val = max(labels)
    for i, l in enumerate(labels):
        if i in centroid_topologies:
            f.write(str(max_val + 1) + "\n")
        else:
            f.write(str(l) + "\n")
