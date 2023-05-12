import numpy as np
import csv
# 生成随机白噪声
num = 16000
wave = np.zeros(num)
for i in range(13000):
    wave[i] = np.random.randn()
dt = 0.02
wave = wave / max(abs(wave)) * 0.3 * 9.8
wave_len = len(wave)
time = np.arange(0, wave_len * dt, dt)
with open("whit_noise.txt", 'w', newline="\n") as f:
    wr = csv.writer(f)
    wr.writerows(zip(wave))

from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_squared_log_error
