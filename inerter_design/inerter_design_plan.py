import numpy as np
import pandas as pd
from ML.rand_vibra import mean_square_resp as resp
from ML.ga.my_GA import AGA


def fitness(S0, wg, xig, w0, xi, gamma_t):
    """

    :param S0:
    :param wg:
    :param xig:
    :param w0:
    :param xi:
    :param gamma_t: 减震比
    :return:
    """

    def fitness_ga(p):
        miu, zeta, kappa = p
        gamma = np.sqrt(resp.inerter(S0, wg, xig, w0, xi, miu, zeta, kappa)) / np.sqrt(resp.normal(S0, wg, xig, w0, xi))
        # 最大耗能变形放大率 α
        # alpha = resp.inerter_in(S0, wg, xig, w0, xi, miu, zeta, kappa) / resp.normal(S0, wg, xig, w0, xi)
        if abs(gamma - gamma_t) > 0.01:
            return 10
        return zeta

    return fitness_ga


if __name__ == '__main__':
    T = np.arange(0.1, 6, 0.1)
    data = {"I0": [31.42, 25.13, 20.94],
            "I1": [25.13, 20.94, 17.95],
            "II": [17.95, 15.71, 13.96],
            "III": [13.96, 11.42, 9.67],
            "IV": [9.67, 8.38, 6.98]
            }
    wg = pd.DataFrame(data, index=["设计地震第1组", "设计地震第2组", "设计地震第3组"])
    site_cate = wg.columns
    group = wg.index
    xig = pd.Series([0.64, 0.64, 0.72, 0.80, 0.90], index=["I0", "I1", "II", "III", "IV"])
    site_cond = pd.DataFrame(columns=["类型编号", "类型名称", "wg", "xig"])
    idx = 0
    for i in range(len(site_cate)):
        for j in range(len(group)):
            type_name = group[j] + "+" + site_cate[i]
            wg_cur = wg[site_cate[i]][j]
            xig_cur = xig[i]
            site_cond.loc[len(site_cond)] = [idx, type_name, wg_cur, xig_cur]
            idx += 1

    xi = np.arange(start=0.02, stop=0.2, step=0.02, dtype=float)
    gamma_t = [0.5, 0.6, 0.7, 0.8, 0.9]
    design_result = pd.DataFrame(
        columns=["周期", "场地编号", "场地名称", "wg", "xig", "结构阻尼比", "减震比", "miu", "zeta", "kappa"])
    for i in T:
        print(i)
        for j in range(len(site_cond.columns)):
            for k in xi:
                for l in gamma_t:
                    site_no_cur = site_cond.iloc[j][0]
                    site_name_cur = site_cond.iloc[j][1]
                    wg_cur = site_cond.iloc[j][2]
                    xig_cur = site_cond.iloc[j][3]
                    T_cur = i
                    w0_cur = 2 * np.pi / T_cur
                    xi_cur = k
                    gamma_t_cur = l
                    func = fitness(1, wg_cur, xig_cur, w0_cur, xi_cur, gamma_t_cur)
                    ga = AGA(func=func, n_dim=3, size_pop=200, max_iter=500, lb=[0, 0, 0], ub=[1, 1, 1],
                             precision=1e-7)
                    best_x, best_y = ga.run()
                    miu = best_x[0]
                    zeta = best_x[1]
                    kappa = best_x[2]
                    design_result.loc[len(design_result)] = [T_cur, site_no_cur, site_name_cur, wg_cur, xig_cur, xi_cur,
                                                             gamma_t_cur, miu, zeta, kappa]
