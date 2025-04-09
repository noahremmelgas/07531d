import numpy as np
import subprocess
import csi_read
import time
import FILA


def run_feitcsi():
    command = [
    "sudo", "feitcsi", "--mode", "measureinject", "--frequency", "5180", 
    "--channel-width", "40", "--format", "HT", "--inject-delay", "10000", "--inject-repeat", "100" 
    "--output-file", "/home/noah/Documents/07531d/07531d/csi.txt"
    ]
    try:
        process = subprocess.run(command)
        while True:
            print("Still running")
            time.sleep(1)

            csi_matrix = csi_read.CSIRead.main('csi.txt')

            FILA.distance.main(csi_matrix)
            
            with open('csi.txt', 'w'):  # Open in write mode ('w') truncates the file
                pass

            

    except ValueError as v:
        while True:
            print(v)
            time.sleep(2) 
            
    except KeyboardInterrupt:
        process.terminate()
        return
    


run_feitcsi()

csi_matrix = csi_read.CSIRead.main('files/csi_3m.txt')
FILA.distance.main(csi_matrix)
csi_matrix = csi_read.CSIRead.main('csi.txt')
FILA.distance.main(csi_matrix)



