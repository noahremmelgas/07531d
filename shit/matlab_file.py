from scipy.io import loadmat
import numpy as np
import h5py

subcarrier_num = 57
bw = 20*10**6
c = 299792458

antenna_loc = np.array([[0, 0, 0],
                        [0.0514665, 0, 0],
                        [0, 0.0514665, 0]])

subcarrier_freq = np.linspace(5.81539e9, 5.8347e9, 57).reshape(1,57)

subcarrier_lambda = c / subcarrier_freq
print(subcarrier_lambda)

est_rco = np.zeros([3,1])


data = loadmat('tests/csi_src_test.mat')
data = data['csi']

#data = np.load('tests/npz_files/loc_minus60deg_3m_reduced.npz')
#data = data['arr_0']

def tof(data, subcarrier_num, bw):
    data = data[:,:,:]
    data_shape = data.shape

    ifft_point = int(np.power(2, np.ceil(np.log2(data_shape[1]))))
    cir_sequence = np.fft.ifft(data, ifft_point, 1)

    cir_sequence = np.squeeze(np.mean(cir_sequence, 3)) #Behövs inte för NPZ filerna

    half_point = int(ifft_point/2)
    half_sequence = cir_sequence[:, 0:half_point, :]

    magnitudes = np.abs(half_sequence)

    peak_indices = np.argmax(magnitudes, axis=1)
    peak_indices = np.squeeze(peak_indices) + 1

    tof_mat = (peak_indices * subcarrier_num) / (ifft_point * bw)

    est_distance = np.mean(tof_mat*c)

    return est_distance

def aoa(data, antenna_loc, est_rco):
    csi_phase = np.unwrap(np.angle(data), 1)

    ant_diff = antenna_loc[:,1:] - antenna_loc[:, 0:1]
    ant_diff_len = np.linalg.norm(ant_diff, axis=0, keepdims=True)
    ant_diff_normalize = ant_diff / ant_diff_len

    phase_diff = csi_phase[:, :, 1:, :] - csi_phase[:, :, 0:1, :] - est_rco[1:,:].reshape(1,1,2); # [T S A-1 L]
    phase_diff = np.unwrap(phase_diff, 1)
    phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi

    cos_mat =  subcarrier_lambda.reshape(1,57,1,1) * phase_diff

    cos_mat = cos_mat / (2 * np.pi * ant_diff_len.reshape(1,1,2))
    cos_mat_mean = np.mean(cos_mat, axis=(1, 3), keepdims=True)
    cos_mat_mean_squeezed = np.squeeze(cos_mat_mean)

    aoa_mat_sol = ant_diff_normalize @ cos_mat_mean_squeezed.T

    print(np.shape(aoa_mat_sol))

    invalid_dim = np.where(sum(ant_diff_normalize, 1) == 0)
    valid_dim = np.setdiff1d([0, 1, 2], invalid_dim)
    # The value of aoa_mat_sol on the invalid dimension is estimated based on the value on the valid dimention.
    invalid_dim = np.where(np.sum(ant_diff_normalize, axis=1) == 0)[0]
    valid_dim = np.setdiff1d(np.array([0, 1, 2]), invalid_dim) #python index starts at 0

    if len(invalid_dim) > 0:
        valid_aoa = aoa_mat_sol[valid_dim, :]
        sum_squared = np.sum(valid_aoa ** 2, axis=0)
        remaining = 1 - sum_squared
        if np.any(remaining < 0):
            remaining[remaining < 0] = 0 #handle negative values due to rounding errors.
        estimated_value = np.sqrt(remaining / len(invalid_dim))
        aoa_mat_sol[invalid_dim, :] = np.tile(estimated_value, (len(invalid_dim), 1))

    aoa_mat = aoa_mat_sol
    return aoa_mat


def main():
    data = loadmat('tests/csi_src_test.mat')
    data = data['csi']
    dist = tof(data, subcarrier_num, bw)

    aoa_mat = aoa(data, antenna_loc, est_rco)
    print(aoa_mat.shape)
    aoa_gt = np.array([[0],[0],[1]])
    error = np.mean(np.arccos(aoa_gt * aoa_mat))

    aoa_gt_transposed = aoa_gt.T

    # Matrix multiplication (equivalent to aoa_gt' * aoa_mat in MATLAB)
    cosines = np.dot(aoa_gt_transposed, aoa_mat)
    # Arccosine (equivalent to acos in MATLAB)
    angles = np.arccos(cosines)

    # Average (equivalent to mean in MATLAB)
    error = np.mean(angles)

    print(f"Angle estimation error: {error}")
    print(dist)

main()