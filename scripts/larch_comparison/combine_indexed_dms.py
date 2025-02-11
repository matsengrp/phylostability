import numpy as np
import sys

base_matrix_name = sys.argv[1]
num_rows = int(sys.argv[2])
num_cols = int(sys.argv[3])
stepsize = int(sys.argv[4])
symmetric = int(sys.argv[5]) > 0
output_filename = sys.argv[6]

dm = np.memmap(output_filename, mode="w+", dtype="float32", shape=(stepsize*num_rows, stepsize*num_cols))

for r in range(num_rows):
    for c in range(num_cols):
        if (not symmetric) or (c > r):
            this_dm = np.memmap(base_matrix_name+"_%d_%d"%(r*stepsize, c*stepsize), mode="r", dtype="float32", shape=(stepsize, stepsize))
            dm[r*stepsize:(r+1)*stepsize, c*stepsize:(c+1)*stepsize] = this_dm[:,:]
        elif symmetric and r == c:
            this_dm = np.memmap(base_matrix_name+"_%d_%d"%(r*stepsize, c*stepsize), mode="r", dtype="float32", shape=(stepsize, stepsize))
            for si in range(stepsize):
                thisrow = r*stepsize + si
                cmin = thisrow + 1
                cmax = (r+1)*stepsize
                dm[thisrow, cmin:cmax] = this_dm[si,0:stepsize-si-1]
            
dm.flush()
