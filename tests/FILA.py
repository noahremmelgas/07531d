import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat 
""" ##### FOR npz files
freq = 5.320e9
subcarrier_num = 30
bandwidth = 40e6
antenna_amount = 3
subcarrier_freq = np.linspace(freq-bandwidth/2, freq+bandwidth/2, subcarrier_num)
####
"""
### For Matlab files
freq = (5.81539e9 + 5.8347e9)/2
subcarrier_num = 57
subcarrier_freq = np.linspace(5.81539e9, 5.8347e9, 57)
c = 299792458
bandwidth = 40e6
####

def csi_to_cir(csi):
    return np.fft.ifft(csi)

def cir_to_csi(csi):
    return np.fft.fft(csi)

def FILA_distance(csi):
    cir = csi_to_cir(csi)
    cir = cir[:,:,0:1]
    cir_shape = np.array([len(cir[:,0,0]),len(cir[0,:,0]),len(cir[0,0,:])])
    
    cirArgMax = np.argmax(np.abs(cir), axis=1, keepdims=True)
    cirMax = np.empty((cir_shape[0], 1, cir_shape[2]),dtype=complex)
    cir_filtered = np.empty((cir_shape[0], cir_shape[1], cir_shape[2]), dtype=complex)


    for i in range(cir_shape[0]):
        for j in range(cir_shape[2]):
            cirMax[i,0,j] = cir[i,cirArgMax[i,0,j],j]
    
    for i in range(cir_shape[0]):
        for j in range(cir_shape[1]):
            for k in range(cir_shape[2]):
                if np.abs(cir[i,j,k]) >= 0.5*np.abs(cirMax[i,0,k]):
                    cir_filtered[i,j,k] = cir[i,j,k] 
                else:
                    cir_filtered[i,j,k] = 0
    
    csi = cir_to_csi(cir_filtered)
    CSI_eff = np.mean(subcarrier_freq / freq * np.abs(cir))

    omega = 200000
    n = 1
    d_LOS = 1/(4*np.pi) * ((c / (bandwidth * np.abs(CSI_eff)))**2 * omega)**(1/n)
    return d_LOS

def main():
    #filenames = np.array([['loc_30deg_1m', 1], ['loc_30deg_2m', 2], ['loc_30deg_3m', 3], ['loc_30deg_4m', 4], ['loc_30deg_5m', 5], 
    #         ['loc_minus60deg_1m', 1], ['loc_minus60deg_2m', 2], ['loc_minus60deg_3m', 3], ['loc_minus60deg_4m', 4], ['loc_minus60deg_5m', 5]])
    #file = filenames[0,0]
    #csi = np.load(f"tests/npz_files/{file}_reduced.npz")['arr_0']

    data = loadmat('tests/csi_src_test.mat')
    csi = data['csi']

    csi = np.mean(csi, axis=3) #For Matlab files, not needed for NPZ

    print(csi.shape)
    print(FILA_distance(csi))

                    

        

main()