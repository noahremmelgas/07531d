import numpy as np

freq = 5.320e9
subcarrier_num = 30
bandwidth = 40e6
antenna_amount = 3
subcarrier_freq = np.linspace(freq-bandwidth/2, freq+bandwidth/2, subcarrier_num)

def csi_to_cir(csi):
    return np.fft.ifft(csi)

def cir_to_csi(csi):
    return np.fft.fft(csi)

def main():
    omega = 1
    n = 1
    filenames = np.array([['loc_30deg_1m', 1], ['loc_30deg_2m', 2], ['loc_30deg_3m', 3], ['loc_30deg_4m', 4], ['loc_30deg_5m', 5], 
             ['loc_minus60deg_1m', 1], ['loc_minus60deg_2m', 2], ['loc_minus60deg_3m', 3], ['loc_minus60deg_4m', 4], ['loc_minus60deg_5m', 5]])
    file = filenames[0,0]

    csi = np.load(f"tests/npz_files/{file}_reduced.npz")['arr_0']

    cir = csi_to_cir(csi)
    print(cir.shape)
    cirArgMax = np.argmax(np.abs(cir), axis=1, keepdims=True)
    cirMax = np.empty((30000,1,3),dtype=complex)
    cir_filtered = np.empty((30000,30,3), dtype=complex)
    print(cirArgMax.shape)
    for i in range(len(cir[:,0,0])):
        for j in range(3):
            cirMax[i,0,j] = cir[i,cirArgMax[i,0,j],j]
    
    for i in range(len(cir[:,0,0])):
        for j in range(len(cir[0,:,0])):
            for k in range(len(cir[0,0,:])):
                if cir[i,j,k] >= 0.5*np.abs(cirMax[i,0,k]):
                    cir_filtered[i,j,k] = cir[i,j,k] 
                else:
                    cir_filtered[i,j,k] = 0
    
    csi = cir_to_csi(cir_filtered
                    

            






main()