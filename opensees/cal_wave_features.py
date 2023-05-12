import re
import numpy as np
import pandas as pd
from numba import jit


@jit(nopython=True)
def Newmark(K, M, C, ag, dt, E):
    """
        利用Newmarkβ计算结构的时程响应
        :param K: 刚度矩阵
        :param M: 质量矩阵
        :param C: 阻尼矩阵
        :param ag: 加速度时程
        :param dt: 时间间隔
        :param E: 单位向量
        :return: 结构的时程信息
        """
    length = ag.size
    dof = E.shape[0]
    y = np.zeros((dof, length), dtype=float)
    dy = np.zeros((dof, length), dtype=float)
    ddy = np.zeros((dof, length), dtype=float)
    ddy_ab = np.zeros((dof, length), dtype=float)

    for i in range(1, length - 1):
        ddy_ab[..., i] = ddy[..., i] + ag[i]
        z_acc = ag[i + 1] - ag[i]
        z_y1 = np.linalg.inv(K + 2 * C / dt + M * 4 / (dt ** 2))
        z_y2 = -z_acc * np.dot(M, E) + np.reshape((4 / dt) * np.dot(M, dy[..., i]) + 2 * np.dot(M, ddy[..., i]) + 2 * np.dot(C, dy[
            ..., i]), (dof, 1))
        z_y = np.reshape(np.dot(z_y1, z_y2), (dof, ))
        y[..., i + 1] = y[..., i] + z_y
        z_dy = 2 / dt * z_y - 2 * dy[..., i]
        dy[..., i + 1] = dy[..., i] + z_dy
        z_ddy = 4 / (dt ** 2) * z_y - (4 / dt) * dy[..., i] - 2 * ddy[..., i]
        ddy[..., i + 1] = ddy[..., i] + z_ddy
    ddy_ab[..., length - 1] = ddy[..., length - 1] + ag[..., length - 1]
    return [y, dy, ddy, ddy_ab]


def cal_reaction_spectrum(ag, dt, T, xi):
    """
    用于计算反应谱参数
    :param ag: 地震动加速度
    :param dt: 加速度时间间隔
    :param T: 基本周期
    :param xi: 结构阻尼比
    :return: [SA: 加速度反应谱值, SV: 速度反应谱值, SD: 位移反应谱值]
    """
    m = 1
    w = np.divide(2 * np.pi, T)
    k = w ** 2 * m
    c = 2 * xi * np.sqrt(m * k)
    M0 = np.array(m, ndmin=2, dtype=float)
    C0 = np.array(c, ndmin=2, dtype=float)
    K0 = np.array(k, ndmin=2, dtype=float)
    E = np.array(1.0, ndmin=2, dtype=float)
    [y, dy, ddy, ddy_ab] = Newmark(K0, M0, C0, ag, dt, E)
    SA = np.max(abs(ddy_ab))
    SV = np.max(abs(dy))
    SD = np.max(abs(y))
    return [SA, SV, SD]


def cal_peek(ag, dt):
    """
    计算峰值加速度、速度和位移(PGA, PGV, PGD)
    :param ag: 地震动加速度
    :param dt: 加速度时间间隔
    :return: [PGA: 峰值加速度, PGV: 峰值速度, PGD: 峰值位移]
    """
    length = np.size(ag)
    # 计算速度
    vg = np.zeros(length)
    for i in range(1, length):
        vg[i] = vg[i - 1] + ag[i - 1] * dt
    # 计算位移
    dg = np.zeros(length)
    for i in range(1, length):
        dg[i] = dg[i - 1] + vg[i - 1] * dt
    PGA = np.max(abs(ag))
    PGV = np.max(abs(vg))
    PGD = np.max(abs(dg))
    return [PGA, PGV, PGD]


def cal_reaction_spectrum_max(ag, dt, T_start, T_end, xi):
    """
    用于计算反应谱峰值指标
    :param ag: 地震动加速度
    :param dt: 加速度时间间隔
    :param T_start: 开始周期
    :param T_end: 结束周期
    :param xi: 结构阻尼比
    :return: [SA_max: 加速度反应谱值峰值, SV_max: 速度反应谱值峰值, SD_max: 位移反应谱值峰值]
    """
    T = np.arange(T_start, T_end, 0.01)
    length = np.size(T)
    SA = np.zeros(length)
    SV = np.zeros(length)
    SD = np.zeros(length)
    for i in range(length):
        SA[i], SV[i], SD[i] = cal_reaction_spectrum(ag, dt, T[i], xi)
    SA_max = np.max(SA)
    SV_max = np.max(SV)
    SD_max = np.max(SD)
    return [SA_max, SV_max, SD_max]


def cal_reaction_spectrum_avg(ag, dt, T_start, T_end, xi):
    """
    用于计算反应谱均值指标
    :param ag: 地震动加速度
    :param dt: 加速度时间间隔
    :param T_start: 开始周期
    :param T_end: 结束周期
    :param xi: 结构阻尼比
    :return: [SA_avg: 加速度反应谱均值, SV_avg: 速度反应谱值均值, SD_avg: 位移反应谱值均值]
    """
    T = np.arange(T_start, T_end, 0.01)
    length = np.size(T)
    SA = np.zeros(length)
    SV = np.zeros(length)
    SD = np.zeros(length)
    for i in range(length):
        SA[i], SV[i], SD[i] = cal_reaction_spectrum(ag, dt, T[i], xi)
    SA_avg = np.sum(SA) / length
    SV_avg = np.sum(SV) / length
    SD_avg = np.sum(SD) / length
    return [SA_avg, SV_avg, SD_avg]


def cal_effective_peek(ag, dt):
    """
    计算有效地震动峰值(EPA, EPV, EPD)
    :param ag: 地震动加速度
    :param dt: 加速度时间间隔
    :return: [EPA: 加速度有效峰值, EPV: 速度有效峰值, EPD: 位移有效峰值]
    """
    [EPA,tmp, tmp] = cal_reaction_spectrum_avg(ag, dt, 0.1, 0.5, 0.05)
    [tmp, EPV, tmp] = cal_reaction_spectrum_avg(ag, dt, 0.8, 2.0, 0.05)
    [tmp, tmp, EPD] = cal_reaction_spectrum_avg(ag, dt, 2.5, 4.0, 0.05)
    EPA = EPA / 2.5
    EPV = EPV / 2.5
    EPD = EPD / 2.5
    return [EPA, EPV, EPD]


def cal_energy_duration(ag, dt, start_energy_percent, end_energy_percent):
    """
    用于计算能量持时的方法
    :param ag: 地震动加速度
    :param dt: 加速度时间间隔
    :param start_energy_percent: 开始的能量百分比(5 => 5%)
    :param end_energy_percent: 结束的能量百分比(95 => 95%)
    :return: [td: 对应的持时, l: 开始能量比对应的时间的索引, r: 结束能量比对应的时间的索引]
    """
    length = np.size(ag)
    g = 9.8
    e = np.pi / (2 * g) * ag ** 2
    e_all = np.sum(e) * dt
    l = 0
    r = 0
    e_s = e_all * start_energy_percent * 0.01
    e_e = e_all * end_energy_percent * 0.01
    e_acc = 0
    for i in range(length):
        if e_acc >= e_s:
            l = i
            break
        e_acc = e_acc + e[i] * dt
    for i in range(l, length):
        if e_acc >= e_e:
            r = i
            break
        e_acc = e_acc + e[i] * dt
    td = (r - l) * dt
    return [td, l, r]


def cal_Ic(ag, dt):
    """
    计算Park-Ang指标
    :param ag: 地震动加速度
    :param dt: 加速度时间间隔
    :return: Ic: Park-Ang指标
    """
    [td, l, r] = cal_energy_duration(ag, dt, 0.05, 0.95)
    index_l = l
    index_r = r
    Pa = 0
    for i in range(index_l, index_r + 1):
        Pa = Pa + ag[i] ** 2 * dt
    Ic = np.sqrt(Pa) ** 1.5 * td ** 0.5
    return Ic


def cal_Housner(ag, dt):
    """
    计算地震动的Housner强度
    :param ag: 地震动加速度
    :param dt: 加速度时间间隔
    :return: [Pa: 加速度Housner强度, Pv: 速度Housner强度, Pd: 位移Housner强度]
    """
    length = np.size(ag)
    # 计算速度
    vg = np.zeros(length)
    for i in range(1, length):
        vg[i] = vg[i - 1] + ag[i - 1] * dt
    # 计算位移
    dg = np.zeros(length)
    for i in range(1, length):
        dg[i] = dg[i - 1] + vg[i - 1] * dt
    [td, l, r] = cal_energy_duration(ag, dt, 0.05, 0.95)
    index_l = l
    index_r = r
    Pa = 0
    Pv = 0
    Pd = 0
    for i in range(index_l, index_r + 1):
        Pa = Pa + ag[i] ** 2 * dt
        Pv = Pv + vg[i] ** 2 * dt
        Pd = Pd + dg[i] ** 2 * dt
    Pa = Pa / td
    Pv = Pv / td
    Pd = Pd / td
    return [Pa, Pv, Pd]

def get_all_features(ag, dt, xi):
    """
    计算所有的地震波特征参数
    :param ag: 地震动加速度
    :param dt: 时间间隔
    :param xi: 结构阻尼比
    :return: 所有的地震波特征参数
    """
    # 计算0.1~3s的反应谱参数
    T = np.arange(0.1, 3.1, 0.1)
    T_len = np.size(T)
    SA = np.zeros(T_len)
    SV = np.zeros(T_len)
    SD = np.zeros(T_len)
    for i in range(T_len):
        SA[i], SV[i], SD[i] = cal_reaction_spectrum(ag, dt, T[i], xi)
    # 计算反应谱均值指标0.1~3s
    SA_avg, SV_avg, SD_avg = cal_reaction_spectrum_avg(ag, dt, 0.01, 3, xi)
    SA_max, SV_max, SD_max = cal_reaction_spectrum_max(ag, dt, 0.01, 3, xi)
    PGA, PGV, PGD = cal_peek(ag, dt)
    EPA, EPV, EPD = cal_effective_peek(ag, dt)
    Pa, Pv, Pd = cal_Housner(ag, dt)
    Ic = cal_Ic(ag, dt)
    return [SA, SV, SD, SA_avg, SV_avg, SD_avg, SA_max, SV_max, SD_max, PGA, PGV, PGD, EPA, EPV, EPD, Pa, Pv, Pd, Ic]


if __name__ == "__main__":
    w0 = 2 * np.pi / 0.56
    k0 = 192 * 1e6
    m0 = 1500 * 1e3
    xi = 0.05
    c0 = xi * 2 * m0 * w0
    filename = "./waves_test/" + "RSN8164_DUZCE_487-NS.AT2";
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

    input = pd.DataFrame(data=None,
                              columns=["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "k1", "k2", "k3", "k4",
                                       "k5", "k6", "k7", "k8", "k9", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8",
                                       "c9", "sa1", "sa2", "sa3", "sa4", "sa5", "sa6", "sa7", "sa8", "sa9", "sa10",
                                       "sa11", "sa12", "sa13", "sa14", "sa15", "sa16", "sa17", "sa18", "sa19", "sa20",
                                       "sa21", "sa22", "sa23", "sa24", "sa25", "sa26", "sa27", "sa28", "sa29", "sa30",
                                       "sv1", "sv2", "sv3", "sv4", "sv5", "sv6", "sv7", "sv8", "sv9", "sv10", "sv11",
                                       "sv12", "sv13", "sv14", "sv15", "sv16", "sv17", "sv18", "sv19", "sv20", "sv21",
                                       "sv22", "sv23", "sv24", "sv25", "sv26", "sv27", "sv28", "sv29", "sv30", "sd1",
                                       "sd2", "sd3", "sd4", "sd5", "sd6", "sd7", "sd8", "sd9", "sd10", "sd11", "sd12",
                                       "sd13", "sd14", "sd15", "sd16", "sd17", "sd18", "sd19", "sd20", "sd21", "sd22",
                                       "sd23", "sd24", "sd25", "sd26", "sd27", "sd28", "sd29", "sd30", "sa_max",
                                       "sa_avg", "sv_max", "sv_avg", "sd_max", "sd_avg", "pga", "pgv", "pgd", "epa",
                                       "epv", "epd", "pa", "pv", "pd", "ic"])
