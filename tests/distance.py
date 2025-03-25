import numpy as np
import matplotlib.pyplot as plt

#Constants
f_k = 5.160 * 10**9
f_c = 5 * 10**9
c = 299792458
num_subcarriers = 30

def csi_to_cir(csi):
  cir = np.fft.ifft(csi)
  return cir

#Example with windowing to reduce sidelobes.
def csi_to_windowed_cir(csi, window=None):
    if window is not None:
        if len(window) != len(csi):
            raise ValueError("Window length must match CSI length.")
        csi_windowed = csi * window
        cir = np.fft.ifft(csi_windowed)
        return cir

    else:
        return csi_to_cir(csi)
    
def distance(csi:list, omega:int, n:int):
    window = np.hamming(num_subcarriers)
    windowed_cir = csi_to_windowed_cir(csi, window)
    subcarrier = np.argmax(np.abs(windowed_cir))
    max_value = np.abs(windowed_cir[subcarrier])

    filtered_cir = np.empty(windowed_cir.shape, dtype=complex)
    for k in range(len(windowed_cir)):
        if np.abs(windowed_cir[k]) <= 0.5*max_value:
            filtered_cir[k] = 0
        else:
            filtered_cir[k] = windowed_cir[k]

    CSI_eff = ((f_c*(subcarrier*20)/f_k*np.abs(filtered_cir)))

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
        for k in range(3):
            csi = csi_load[0:10000,0:30,k]
            d_LOS_array = np.array([])
            for i in csi:
                i = np.ndarray.flatten(i)
                d_LOS, csi_EFF = distance(i, omega, n)
                d_LOS_array=np.append(d_LOS_array, d_LOS)
            distance_array = np.append(distance_array, np.mean(d_LOS_array))
        print(np.mean(distance_array))  



main()


