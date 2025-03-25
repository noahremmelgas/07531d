import numpy as np
import h5py
f = h5py.File('csi_src_test.mat','r')
data = f.get('csi')
data = np.array(data) # For converting to a NumPy array
print(data)