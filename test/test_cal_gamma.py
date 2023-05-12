"""
计算真实减震比
"""
import math

from ML.rand_vibra import mean_square_resp as resp

miu, zeta, kappa = [0.31386, 0.00057, 0.02964]

S0 = 8.81
wg = 31.42000
xig = 0.64000
w0 = 2 * math.pi / 0.100
xi = 0.02000

gamma = math.sqrt(resp.inerter(S0, wg, xig, w0, xi, miu, zeta, kappa)) / math.sqrt(resp.normal(S0, wg, xig, w0, xi))