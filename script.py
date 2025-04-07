import numpy as np
import os
import subprocess
import csi_read
import time


def run_feitcsi():
    while True:
        try:
            process = subprocess.Popen(['yes', ' "Hello World!"'], stdout=subprocess.PIPE, text=True)
            line = process.stdout.readlines()
            print(line.strip())
        
            process.terminate()

            csi_matrix = csi_read.CSIRead.main()

            print(csi_matrix.shape)
        except KeyboardInterrupt:
            break

file_name = 'files/csi_3m.txt'
csi_matrix = csi_read.CSIRead.main(file_name)

print(csi_matrix.shape)



