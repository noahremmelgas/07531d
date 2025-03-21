file_path = './csi/use_in_paper/2_objects/test/D=2020-05-20_T=18-03-48--fullbottle05.dat'

import pandas as pd
from os import path
from struct import unpack
from numpy import zeros, array
from numba import njit
import numpy as np

@njit(cache=True)
def __signbit_convert(data: int) -> int:
    if data & 512:
        data -= 1024
    return data

@njit(cache=True)
def _read_csi_native(local_h: int, nr: int, nc: int, num_tones: int) -> list:
    csi_re = zeros((nr * nc, num_tones))
    csi_im = zeros((nr * nc, num_tones))

    BITS_PER_BYTE = 8
    BITS_PER_SYMBOL = 10
    bits_left = 16

    h_data = local_h[0] + (local_h[1] << BITS_PER_BYTE)
    current_data = h_data & 65535
    idx = 2

    for k in range(num_tones):
        for nc_idx in range(nc):
            for nr_idx in range(nr):
                if bits_left < BITS_PER_SYMBOL:
                    h_data = local_h[idx] + (local_h[idx + 1] << BITS_PER_BYTE)
                    idx += 2
                    current_data += h_data << bits_left
                    bits_left += 16
                
                imag = current_data & 1023
                bits_left -= BITS_PER_SYMBOL
                current_data = current_data >> BITS_PER_SYMBOL

                if bits_left < BITS_PER_SYMBOL:
                    h_data = local_h[idx] + (local_h[idx + 1] << BITS_PER_BYTE)
                    idx += 2
                    current_data += h_data << bits_left
                    bits_left += 16

                real = current_data & 1023
                bits_left -= BITS_PER_SYMBOL
                current_data = current_data >> BITS_PER_SYMBOL

                csi_re[nr_idx + nc_idx * 2, k] = __signbit_convert(real)
                csi_im[nr_idx + nc_idx * 2, k] = __signbit_convert(imag)

    return csi_re, csi_im

def __read_csi(csi_buf: list, nr: int, nc: int, num_tones: int) -> dict:
        csi_re, csi_im = _read_csi_native(csi_buf, nr, nc, num_tones)
        return array([csi_re[i] + 1j * csi_im[i] for i in range(nr * nc)])

raw = []

with open(file_path, 'rb') as f:
            len_file = path.getsize(file_path)
            cur = 0

            while cur < len_file:
                csi_matrix = {}
                csi_matrix['field_len'] = unpack('>H', f.read(2))[0] # field_len doesn`t use
                csi_matrix['timestamp'], csi_matrix['csi_len'], csi_matrix['tx_channel'], csi_matrix['err_info'], csi_matrix['noise_floor'], csi_matrix['rate'],  csi_matrix['bandWitdh'], csi_matrix['num_tones'],  csi_matrix['nr'], csi_matrix['nc'], csi_matrix['rssi0'], csi_matrix['rssi1'], csi_matrix['rssi2'], csi_matrix['rssi3'], csi_matrix['payload_len'] = unpack('>QHHBBBBBBBBBBBH', f.read(25))
               
                if csi_matrix['csi_len']:
                    buf = unpack('B' * csi_matrix['csi_len'], f.read(csi_matrix['csi_len']))
                    csi_matrix['csi'] = __read_csi(buf, csi_matrix['nr'], csi_matrix['nc'], csi_matrix['num_tones'])
                else:
                    csi_matrix['csi'] = [] 
                               
                csi_matrix['payload'] = unpack('B' * csi_matrix['payload_len'], f.read(csi_matrix['payload_len']))
                cur += 27 + csi_matrix['csi_len'] + csi_matrix['payload_len']
                
                raw.append(csi_matrix)

#f = open('mytext.txt','w')
#f.write(str(raw))
df = pd.DataFrame(raw)
#df.to_excel("output.xlsx",
             #sheet_name='Sheet_name_1')

rssi0 = df.rssi0.iloc[1]
print(rssi0)
N=3
Distance = 2.5*10**((75 - rssi0)/(10*N))
print(Distance)
    