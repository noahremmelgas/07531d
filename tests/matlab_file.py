from scipy.io import loadmat
import numpy as np
import h5py

subcarrier_num = 57
bw = 20*10**6
c = 299792458
data = loadmat('tests/csi_src_test.mat')
data = data['csi']

data = np.load('tests/npz_files/loc_30deg_4m_reduced.npz')
data = data['arr_0']

data_shape = data.shape

ifft_point = int(np.power(2, np.ceil(np.log2(data_shape[1]))))
print(ifft_point)
cir_sequence = np.fft.ifft(data, ifft_point, 1)

#cir_sequence = np.squeeze(np.mean(cir_sequence, 3)) #Behövs inte för NPZ filerna

print(cir_sequence.shape)

half_point = int(ifft_point/2)

half_sequence = cir_sequence[:, 0:half_point, :]

magnitudes = np.abs(half_sequence)

peak_indices = np.argmax(half_sequence, axis=1)

peak_indices = np.squeeze(peak_indices) + 1

print(peak_indices)

tof_mat = (peak_indices * subcarrier_num) / (ifft_point * bw)

print(tof_mat)

est_distance = np.mean(tof_mat*c)

print(est_distance)