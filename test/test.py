import math

import numpy as np

from ML.rand_vibra import fre_resp_func as fre
from ML.rand_vibra import spectral_density


def test_spectral_density():
    """
    II类场地， 设计地震分组为三组，8度设防
    """
    S0 = 9.50
    wg = 13.96
    xig = 0.72
    w = np.arange(0, 1000, 0.01)
    SA = spectral_density.cal_spectral_density(w, wg, xig, S0)
    print(SA)


def test_fre_resp_func():
    w = 100
    s = w * 1j
    w0 = 2 * math.pi / 0.56
    xi = 0.05
    miu = 0.0298
    zeta = 0.0016
    kappa = 0.0301
    print(fre.normal(s, xi, w0))
    print(fre.inerter(w, xi, w0, miu, zeta, kappa))


def cal_mean_square_value():
    S0 = 9.50
    wg = 13.96
    xig = 0.72
    step = 0.5
    w = np.arange(0, 20, step)
    s = w * -1j
    w0 = 2 * math.pi / 0.56
    xi = 0.05
    miu = 0.0298
    zeta = 0.0016
    kappa = 0.0301

    # k = 192 * 10 ** 6
    # m = 1500 * 10 ** 3
    # mr = miu * m
    # kd = kappa * k
    # cd = zeta * 2 * m * w0

    SA = spectral_density.cal_spectral_density(w, wg, xig, S0)
    H0 = fre.normal(s, xi, w0)
    H_in = fre.inerter(w, xi, w0, miu, zeta, kappa)
    y0 = abs(H0) ** 2 * SA
    y_in = abs(H_in) ** 2 * SA
    sigma0 = sum(y0 * step)
    sigma_in = sum(y_in * step)
    print(sigma0)
    print(sigma_in)



if __name__ == '__main__':

    # test_spectral_density()
    # test_fre_resp_func()
    cal_mean_square_value()

