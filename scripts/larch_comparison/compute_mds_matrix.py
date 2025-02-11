import sys
from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa
import numpy as np


input_filename = sys.argv[1]
dim_filename = sys.argv[2]
output_filename = sys.argv[3]

with open(dim_filename, "r") as f:
    dim = tuple([int(x) for x in f.readlines()[0].split()])

input_data = np.memmap(input_filename, dtype='float32', mode='r+', shape=dim)
np.nan_to_num(input_data, copy=False)
result = pcoa(DistanceMatrix(input_data[:,:] + input_data[:,:].T), number_of_dimensions=2)
output_data = np.memmap(output_filename, dtype='float32', mode='w+', shape=(input_data.shape[0], 2))
output_data[:,0] = result.samples["PC1"]
output_data[:,1] = result.samples["PC2"]
output_data.flush()
