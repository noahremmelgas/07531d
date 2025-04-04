from CSIKit.reader import get_reader
from CSIKit.util import csitools
import numpy as np

my_reader = get_reader('tests/measurement.txt')
csi_data = my_reader.read_file('tests/measurement.txt')

csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data)
print(csi_matrix)

np.savez("tests/csi_data.npz", csi_matrix)

f = np.load('tests/csi_data.npz')
print(f['arr_0'])