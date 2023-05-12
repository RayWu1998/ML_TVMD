import numpy as np
from common_function import newmark
import os
import pandas as pd
import re


if __name__ == '__main__':
    w0 = 2 * np.pi / 0.56
    k0 = 192 * 1e6
    m0 = 1500 * 1e3
    xi = 0.05
    c0 = xi * 2 * m0 * w0
    M0 = np.array(m0, ndmin=2, dtype=float)
    C0 = np.array(c0, ndmin=2, dtype=float)
    K0 = np.array(k0, ndmin=2, dtype=float)
    E0 = np.array(1, ndmin=2, dtype=float)
    wave = 0
    dt = 0
    for root, dirs, files in os.walk("./waves", topdown=False):
        for name in files:
            print(name)
            wave_name = name.split('.')[0]
            if name.endswith(".AT2"):
                filename = os.path.join(root, name)
                with open(filename, 'r') as fo:
                    for i in range(3):
                        fo.readline()
                    line = fo.readline()
                    target = re.search(r'DT= *(.*) *SEC', line, re.I | re.M)
                    dt = float(target.group(1))
                    wave = pd.read_table(filename, skiprows=4, sep="  ", header=None, engine='python')
                    wave = wave.fillna(0)
                    wave = wave.values
                    size = wave.size
                    wave = wave.reshape((size,))
                    wave = wave / max(abs(wave)) * 1.4
            if name.endswith(".txt"):
                filename = os.path.join(root, name)
                wave = np.loadtxt(filename)
                wave = wave / max(abs(wave)) * 1.4
                dt = 0.02
    resp0 = newmark(K0, M0, C0, wave, dt, E0)
