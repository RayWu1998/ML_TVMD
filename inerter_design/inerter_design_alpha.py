from ML.rand_vibra import mean_square_resp as resp
import numpy as np
import math


def fitness(S0, wg, xig, w0, xi, gamma_t, h):
    """
    :param S0:
    :param wg:
    :param xig:
    :param w0:
    :param xi:
    :param gamma_t: 减震比
    :param h: 惩罚权重
    :return:
    """

    def fitness_ga(p):
        miu, zeta, kappa = p
        gamma = np.sqrt(resp.inerter(S0, wg, xig, w0, xi, miu, zeta, kappa)) / np.sqrt(resp.normal(S0, wg, xig, w0, xi))
        # 最大耗能变形放大率 α
        alpha = np.sqrt(resp.inerter_in(S0, wg, xig, w0, xi, miu, zeta, kappa)) / np.sqrt(
            resp.normal(S0, wg, xig, w0, xi))
        f = -alpha
        # 如果超过了目标减震比要求，则f要>0
        if abs(gamma - gamma_t) > 0.01:
            f = -alpha + 100 * abs(gamma - gamma_t)
        # f = -alpha + 10 * abs(gamma - gamma_t)
        return f

    return fitness_ga


from ML.ga.my_GA import AGA

if __name__ == '__main__':
    # II类场地，设计地震第二组，特征周期Tg=0.4s，多遇地震,8度设防
    S0 = 1
    wg = 15.71
    xig = 0.72
    w0 = 2 * math.pi / 3.157
    xi = 0.158
    func = fitness(S0, wg, xig, w0, xi, 0.7, 1)
    # ga = GA(func=func, n_dim=3, size_pop=200, max_iter=2000, prob_mut=0.001, lb=[0, 0, 0], ub=[1, 1, 1], precision=1e-7)
    ga = AGA(func=func, n_dim=3, size_pop=200, max_iter=1000, lb=[0, 0, 0], ub=[1, 1, 1], precision=1e-8, early_stop=500)
    best_x, best_y = ga.run()
    print('best_x:', best_x, '\n', 'best_y:', best_y)
    miu = best_x[0]
    zeta = best_x[1]
    kappa = best_x[2]
    gamma = np.sqrt(resp.inerter(1, wg, xig, w0, xi, miu, zeta, kappa)) / np.sqrt(
        resp.normal(1, wg, xig, w0, xi))
    print("gamma:", gamma)

    import pandas as pd
    import matplotlib.pyplot as plt

    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    plt.show()

