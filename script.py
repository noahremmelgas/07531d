import numpy as np
import subprocess
import csi_read
import time
import FILA
import pandas as pd
import os


def run_feitcsi():
    with open('/home/wifi/csi.txt', 'w') as f:  # Open in write mode ('w') truncates the file
                f.truncate(0)
    command = "sudo feitcsi --mode measure --frequency 5180 --channel-width 40 --format HT --output-file /home/wifi/csi.txt"
    process = subprocess.run(["gnome-terminal", "--", "bash", "-c", f"{command}; bash"], check=True)
    time.sleep(2)
    try:

        while True:
            for i in range(1):
                time.sleep(1)

                csi_matrix = csi_read.CSIRead.main('/home/wifi/csi.txt')

                FILA.distance.main(csi_matrix)
            with open('/home/wifi/csi.txt', 'w') as f:  # Open in write mode ('w') truncates the file
                f.truncate(0)
            

            

    except ValueError as v:
        while True:
            print(v)
            time.sleep(2) 
            
    except KeyboardInterrupt:
        process.terminate()
        return
    
    except IndexError:
        process.terminate()
    


#run_feitcsi()

files = ['1dBm', '10dBm', '20dBm'] #, 'csi_5m.txt', 'csi_10m.txt', 'csi_15m.txt', 'csi_20m.txt']

for i in files:
    csi_matrix = csi_read.CSIRead.main(f'/home/noah/Downloads/{i}.txt')
    x = np.load(f"/home/noah/{i}.npy", csi_matrix)
    print(x)
    #print(i)
    #print(np.shape(csi_matrix))
    #FILA.distance.main(csi_matrix[:100,:,:])


"""
csi_matrix = csi_read.CSIRead.main('/home/wifi/csi.txt')

k=0
j=[]
for i in csi_matrix[0,:,0]:
    if np.abs(i) == 0:
        j.append(k)
    k+=1
print(j)
"""
