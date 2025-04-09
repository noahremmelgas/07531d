import numpy as np
import subprocess
import csi_read
import time
import FILA


def run_feitcsi():
    command = [
    "sudo", "feitcsi", "--mode", "measure", "--frequency", "5180", 
    "--channel-width", "20", "--format", "HT",
    "--output-file", "/home/noah/Documents/07531d/07531d/csi.txt"
    ]
    try:
        while True:
            with open('csi.txt', 'w') as f:  # Open in write mode ('w') truncates the file
                f.truncate(0)
            print("Still running")
            for i in range(10):
                time.sleep(1)

                csi_matrix = csi_read.CSIRead.main('csi.txt')

                FILA.distance.main(csi_matrix)
            

            

    except ValueError as v:
        while True:
            print(v)
            time.sleep(2) 
            
    except KeyboardInterrupt:
        return
    


run_feitcsi()

csi_matrix = csi_read.CSIRead.main('files/csi_9m_0d.txt')
FILA.distance.main(csi_matrix)



