import os
import re
import pandas as pd
for root, dirs, files in os.walk("./waves", topdown=False):
    for name in files:
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
                print(wave)

with open('./wave_name.txt', 'r') as fo:
    wave_names = fo.readlines()
    for i in range(len(wave_names)):
        wave_names[i] = wave_names[i].strip('\n')
