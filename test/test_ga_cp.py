import numpy as np


def schaffer(p):
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(x)) - 0.5) / np.square(1 + 0.001 * x)


from ML.ga.my_GA import AGA
from ML.ga.my_GA import GA

# ga = AGA(func=schaffer, n_dim=2, size_pop=100, max_iter=800, lb=[-1, -1], ub=[1, 1], precision=1e-7)
ga = GA(func=schaffer,prob_crossover=0.2, prob_mut=0.4, n_dim=2, size_pop=200, max_iter=800, lb=[-1, -1], ub=[1, 1], precision=1e-7)
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)

import pandas as pd
import matplotlib.pyplot as plt

Y_history = pd.DataFrame(ga.all_history_Y)
fig = plt.figure(figsize=(12, 10))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 每个个体的解
plt.subplot(2, 1, 1)
plt.xlabel("种群迭代次数")
plt.ylabel("解")
plt.plot(Y_history.index, Y_history.values, '.', color='red')

plt.title("种群表现")
# 每代最优解
plt.subplot(2, 1, 2)
plt.xlabel("种群迭代次数")
plt.ylabel("最优解")
Y_history.min(axis=1).cummin().plot(kind='line')
plt.title("最优个体表现")
plt.show()

