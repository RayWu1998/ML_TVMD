def normal(s, xi, w0):
    """
    :param s: s = jw
    :param xi: 固有阻尼比
    :param w0: 单自由度结构的自振圆频率
    :return: 计算出的频率响应函数值
    """
    Hs = -1 / (s ** 2 + 2 * xi * w0 * s + w0 ** 2)
    return Hs


def inerter(w, xi, w0, miu, zeta, kappa):
    """

    :param w: 圆频率
    :param xi: 固有阻尼比
    :param w0: 单自由度结构的自振圆频率
    :param miu: 惯质比
    :param zeta: 名义阻尼比
    :param kappa: 刚度比
    :return: 计算出的频率响应函数值
    """
    i = 1j
    A = miu * w ** 2 - 2 * i * zeta * w0 * w - kappa * w0 ** 2
    B = miu * w ** 4 - 2 * i * w0 * w ** 3 * (miu * xi + zeta) - w0 ** 2 * w ** 2 * (kappa + miu + kappa * miu + 4 * xi * zeta) + \
        2 * i * w0 ** 3 * w * (kappa * xi + kappa * zeta + zeta) + w0 ** 4 * kappa
    Hs = A / B
    return Hs


def inerter_in(w, xi, w0, miu, zeta, kappa):
    """
    求惯容元件位移均方值
    :param w:
    :param xi:
    :param w0:
    :param miu:
    :param zeta:
    :param kappa:
    :return:
    """
    Hs = inerter(w, xi, w0, miu, zeta, kappa)
    s = w * 1j
    Hs_in = (miu * s ** 2 + 2 * zeta * w0 * s) / (miu * s ** 2 + 2 * zeta * w0 * s + kappa * w0 ** 2) * Hs
    return Hs_in
