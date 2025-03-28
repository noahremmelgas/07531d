import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

#Constants
f_k = 4.72 * 10**9
f_c = 5.320 * 10**9
c = 299792458
num_subcarriers = 57

def csi_to_cir(csi):
  cir = np.fft.ifft(csi)
  return cir

def csi_to_windowed_cir(csi, window=None):
    csi_windowed = csi * 1
    cir = np.fft.ifft(csi_windowed)
    return cir

def distance(csi:list, omega:int, n:int):
    window = np.hamming(num_subcarriers)
    windowed_cir = csi_to_windowed_cir(csi, 1)
    subcarrier = np.argmax(np.abs(windowed_cir))
    max_value = np.abs(windowed_cir[subcarrier])

    subcarrier_list = np.arange(0,57)

    filtered_cir = np.empty(windowed_cir.shape, dtype=complex)
    for k in range(len(windowed_cir)):
        if np.abs(windowed_cir[k]) <= 0.5*max_value:
            filtered_cir[k] = 0
        else:
            filtered_cir[k] = windowed_cir[k]

    filtered_cfr = np.fft.fft(filtered_cir)
    
    CSI_eff = f_c*(f_k+np.multiply(subcarrier_list, 40))/f_k
    
    CSI_eff = np.abs(np.multiply(CSI_eff, filtered_cfr*window))

    CSI_eff = np.mean(CSI_eff)

    d_LOS = 1/(4*np.pi)*((((c/(f_c*CSI_eff))**2)*omega)**(1/n))

    return d_LOS, CSI_eff

def main():
    omega = 1
    n = 1
    filenames = [['loc_30deg_1m', 1], ['loc_30deg_2m', 2], ['loc_30deg_3m', 3], ['loc_30deg_4m', 4], ['loc_30deg_5m', 5], 
             ['loc_minus60deg_1m', 1], ['loc_minus60deg_2m', 2], ['loc_minus60deg_3m', 3], ['loc_minus60deg_4m', 4], ['loc_minus60deg_5m', 5]]
    for file, label in filenames:
        distance_array = np.array([])
        csi_load = np.load(f"tests/npz_files/{file}_reduced.npz")['arr_0']
        data = loadmat('tests/csi_src_test.mat')
        data = data['csi']
        for k in range(3):
            csi = data[0:100,:,k,1]

            d_LOS_array = np.array([])
            for i in csi:
                i = np.ndarray.flatten(i)
                d_LOS, csi_EFF = distance(i, omega, n)
                d_LOS_array=np.append(d_LOS_array, d_LOS)
            distance_array = np.append(distance_array, np.mean(d_LOS_array))
        print(np.mean(distance_array))  



main()


