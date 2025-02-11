import numpy as np
import sys

dm_file = sys.argv[1]
dm_dim1 = sys.argv[2]
dm_dim2 = sys.argv[3]

data_mat = np.memmap(dm_file, dtype='float32', mode='w+', shape=(int(dm_dim1), int(dm_dim2)))

