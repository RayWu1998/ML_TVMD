import numpy as np


def schaffer(p):
    x1, x2 = p
    res = 20 + x1 ** 2 + x2 ** 2 - 10 * (np.cos(0.5 * np.pi * x2) + np.cos(0.5 * np.pi * x1))
    return res


from ML.ga.my_GA import AGA
from ML.ga.my_GA import GA

ga = AGA(func=schaffer, n_dim=2, size_pop=100, max_iter=800, lb=[-5, -5], ub=[5, 5], precision=1e-7)
# ga = GA(func=schaffer, n_dim=2, prob_mut=0.2, prob_crossover=0.001, size_pop=100, max_iter=800, lb=[-5, -5], ub=[5, 5], precision=1e-7)
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)

import pandas as pd
import matplotlib.pyplot as plt

Y_history = pd.DataFrame(ga.all_history_Y)
fig, ax = plt.subplots(2, 1)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 每个个体的解
plt.subplot(2, 1, 1)
plt.xlabel("种群迭代次数")
plt.ylabel("解")
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')

plt.title("种群表现")
# 每代最优解
plt.subplot(2, 1, 2)
plt.xlabel("种群迭代次数")
plt.ylabel("最优解")
Y_history.min(axis=1).cummin().plot(kind='line')
plt.title("最优个体表现")
plt.show()

