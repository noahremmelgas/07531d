import numpy as np

filenames = [['loc_30deg_1m', 1], ['loc_30deg_2m', 2], ['loc_30deg_3m', 3], ['loc_30deg_4m', 4], ['loc_30deg_5m', 5], 
             ['loc_minus60deg_1m', 1], ['loc_minus60deg_2m', 2], ['loc_minus60deg_3m', 3], ['loc_minus60deg_4m', 4], ['loc_minus60deg_5m', 5]]

for file, label in filenames:
    data = np.load(f"C:/Users/noahr/OneDrive - Chalmers/Programmering/Kandidatprojekt/csi_classification-master/npz_files/{file}.npz")['arr_0'][0:30000]
    np.savez(f"tests/npz_files/{file}_reduced.npz", data)