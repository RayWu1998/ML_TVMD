from openseespy.opensees import *
import numpy as np
import joblib
from cal_wave_features import get_all_features
import pandas as pd
from ML.ga.my_GA import AGA
from sklearn.preprocessing import StandardScaler
import time

main_features_ok = ['sa_avg', 'pgv', 'sv_max', 'ic', 'sv_avg', 'epv', 'sv16', 'epa', 'sa11', 'sd5', 'pga', 'sd11', 'sv18', 'sa5', 'sv3', 'k3', 'k7', 'k8', 'c4', 'c8', 'c7', 'k5', 'm1', 'm3', 'c6', 'm5', 'm4', 'c2', 'c3', 'c5', 'm8', 'c9', 'm7', 'k9', 'm2', 'c1', 'm6', 'k1', 'm9', 'k2', 'k4', 'k6']
main_features_drift_max = ['sv_avg', 'epv', 'sv18', 'sv_max', 'sv21', 'sv22', 'sa_avg', 'sd12', 'sa12', 'sv20', 'sv19', 'sv15', 'sv13', 'pgv', 'sv23', 'k8', 'c8', 'k7', 'k9', 'c7', 'k4', 'c3', 'k1', 'c4', 'c2', 'k2', 'k3', 'm1', 'k6', 'k5', 'm9', 'm3', 'c6', 'm8', 'm4', 'c1', 'm2', 'm6', 'm5', 'c9', 'c5', 'm7']
main_features_drift_avg = ['sv21', 'sv_avg', 'sv20', 'sv18', 'sv19', 'sv22', 'sa18', 'epv', 'sa_avg', 'sd18', 'sa17', 'sd_avg', 'sv23', 'sv_max', 'sv15', 'k2', 'k8', 'k7', 'k4', 'k3', 'k6', 'k5', 'c3', 'k9', 'k1', 'c2', 'c4', 'c5', 'm1', 'm3', 'c8', 'c7', 'c6', 'm8', 'c9', 'm9', 'm7', 'm2', 'm6', 'm5', 'c1', 'm4']
main_features_a_max = ['pga', 'sa_max', 'epa', 'sd1', 'sa1', 'sa_avg', 'sa2', 'sd2', 'pgv', 'sv2', 'sv_max', 'sd4', 'sa4', 'sa3', 'sd3', 'k1', 'k8', 'm9', 'k9', 'k2', 'k5', 'c2', 'm1', 'k4', 'm8', 'k7', 'k6', 'm6', 'm2', 'c7', 'm4', 'c3', 'c5', 'k3', 'c8', 'c9', 'c4', 'm3', 'c6', 'm7', 'm5', 'c1']
main_features_a_avg = ['pga', 'epa', 'sv_max', 'sv7', 'sd6', 'sa6', 'sd7', 'sa7', 'sv10', 'sd8', 'sa8', 'sa18', 'sa_max', 'sv8', 'pd', 'k1', 'c4', 'c2', 'k8', 'k9', 'k2', 'k3', 'c7', 'k7', 'm6', 'k6', 'm1', 'c8', 'm8', 'c9', 'm5', 'c6', 'k5', 'k4', 'c3', 'm7', 'm9', 'm3', 'm4', 'c5', 'm2', 'c1']

# 纤维截面定义
def Wsection(secID, matID, d, bf, tf, tw, nfdw, nftw, nfbf, nftf):
    """
    create a standard W section given the nominal section properties
    :param secID: 截面编号
    :param matID: 材料编号
    :param d: 截面高度
    :param bf: 腹板厚度
    :param tf: 翼缘宽度
    :param tw: 翼缘厚度
    :param nfdw: 沿腹板高度的纤维数量
    :param nftw: 沿腹板厚度的纤维数量
    :param nfbf: 沿翼缘宽度的纤维数量
    :param nftf: 沿翼缘厚度的纤维数量
    :return:
    """
    dw = d - 2 * tf
    y1 = -d / 2
    y2 = -dw / 2
    y3 = dw / 2
    y4 = d / 2

    z1 = -bf / 2
    z2 = -tw / 2
    z3 = tw / 2
    z4 = bf / 2

    section('Fiber', secID)
    patch('quad', matID, nfbf, nftf, y1, z4, y1, z1, y2, z1, y2, z4)
    patch('quad', matID, nftw, nfdw, y2, z3, y2, z2, y3, z2, y3, z3)
    patch('quad', matID, nfbf, nftf, y3, z4, y3, z1, y4, z1, y4, z4)


# 定义惯容系统
def inerterDamper2DX(inode, jnode, cd, md, ks):
    """
    定义TVMD
    :param inode: 连接节点i
    :param jnode: 连接节点j
    :param cd: 阻尼系数
    :param md: 惯容系数
    :param ks: 刚度
    """

    # set nodes
    x1 = nodeCoord(inode, 1)
    y1 = nodeCoord(inode, 2)
    x2 = nodeCoord(jnode, 1)
    y2 = nodeCoord(jnode, 2)
    rigid_beam_len = 100.0  # 刚性杆的长度

    node_start = inode * 100 + jnode * 10
    node(node_start + 1, x1, y1)
    node(node_start + 2, x1, y1)
    node(node_start + 3, x2, y2)
    node(node_start + 4, x1, y1 - rigid_beam_len)
    node(node_start + 5, x1 - 1, y1 - rigid_beam_len)

    fix(node_start + 1, 0, 1, 1)
    fix(node_start + 2, 0, 1, 0)
    fix(node_start + 3, 0, 1, 1)
    fix(node_start + 5, 0, 1, 0)

    equalDOF(inode, node_start + 1, 1)
    equalDOF(inode, node_start + 4, 1)
    equalDOF(jnode, node_start + 3, 1)

    # dashpot
    mat_start = inode * 100 + jnode * 10
    ele_start = inode * 100 + jnode * 10
    uniaxialMaterial('Viscous', mat_start + 1, cd, 1)
    element('zeroLength', ele_start + 1, node_start + 1, node_start + 2, '-mat', mat_start + 1, '-dir', 1)  # 该构件的方向是x向

    # mass
    mass(node_start + 4, 1e-5, 1e-5, rigid_beam_len * rigid_beam_len * md)

    # Spring
    if x1 == x2 and y1 == y2:
        uniaxialMaterial('Elastic', mat_start + 2, ks)
        element('zeroLength', ele_start + 2, node_start + 2, node_start + 3, '-mat', mat_start + 2, '-dir',
                1)  # 弹簧方向是x向
    else:
        spring_len = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
        uniaxialMaterial('Elastic', mat_start + 2, ks * spring_len)
        element('corotTruss', ele_start + 2, node_start + 2, node_start + 3, 1.0, mat_start + 2)

    # geomTransf Linear 1;
    # Rigid beam
    element('elasticBeamColumn', ele_start + 3, node_start + 2, node_start + 4, 1.0e+10, 1.0, 1.0e+12, 1)
    element('elasticBeamColumn', ele_start + 4, node_start + 5, node_start + 4, 1.0e+12, 1.0, 1.0e-8, 1)


def drift_cal(i_node, height):
    """
    计算层间位移角
    :param i_node: i节点的节点标志
    :param height: 层高
    :return:
    """
    # j_node是i_node的下层节点，tag比i_node小10
    j_node = i_node - 10
    drift = (nodeDisp(i_node, 1) - nodeDisp(j_node, 1)) / height
    return drift


def height(node_tag):
    """
    返回节点标志对应的层高
    :param node_tag: 节点标志
    :return: 层高
    """
    if node_tag < 20:
        return 5.4864  # 1层
    else:
        return 3.9624  # 2~9层


def cal_a(a):
    """
    计算层加速度指标
    :param a: 结构各层的加速度时程
    :return: 各层中最大的最大加速度和平均的最大加速度
    """
    a_each_max = np.max(a, 1)
    return [np.max(a_each_max), np.sum(a_each_max) / np.size(a_each_max)]


def cal_drift(drift):
    """
    计算层间位移角指标
    :param drift: 结构各层的层间位移角时程
    :return: 各层中最大的最大位移角和平均的最大位移角
    """
    drift_each_max = np.max(drift, 1)
    return [np.max(drift_each_max), np.sum(drift_each_max) / np.size(drift_each_max)]


def original_model(wave_file, NPTS, dt, PGA):
    """
    构建非线性模型，计算目标参数
    :param wave_file: 地震波文件名
    :param NPTS: 地震波的记录节点数量
    :param dt: 地震波的时间间隔
    :param PGA: 地震波的放大系数
    :return: [T_1, a_max, a_avg, drift_max, drift_avg]
    """
    wipe()
    model('basic', '-ndm', 2, '-ndf', 3)

    # 框架节点
    h1 = 5.4864  # 1层层高
    h2 = 3.9624  # 2-9层层高
    span_len = 9.144  # 跨长

    for i in range(0, 10):
        for j in range(1, 7):
            if i == 0 or i == 1:
                h = i * h1
            else:
                h = h1 + (i - 1) * h2
            node_tag = 10 * i + j
            x = span_len * (j - 1)
            y = h
            node(node_tag, x, y)

    # 节点约束
    for i in range(0, 10):
        for j in range(1, 7):
            if i == 0:
                fix(i * 10 + j, 1, 1, 1)
            else:
                fix(i * 10 + j, 0, 0, 0)

    # 变截面节点
    bh1 = 1.8288 + h1
    bh2 = 1.8288 + h1 + 2 * h2
    bh3 = 1.8288 + h1 + 4 * h2
    bh4 = 1.8288 + h1 + 6 * h2

    for i in range(1, 7):
        node_tag = 100 + i
        x = span_len * (i - 1)
        y = bh1
        node(node_tag, x, y)
        fix(node_tag, 0, 0, 0)

    for i in range(1, 7):
        node_tag = 200 + i
        x = span_len * (i - 1)
        y = bh2
        node(node_tag, x, y)
        fix(node_tag, 0, 0, 0)

    for i in range(1, 7):
        node_tag = 300 + i
        x = span_len * (i - 1)
        y = bh3
        node(node_tag, x, y)
        fix(node_tag, 0, 0, 0)

    for i in range(1, 7):
        node_tag = 400 + i
        x = span_len * (i - 1)
        y = bh4
        node(node_tag, x, y)
        fix(node_tag, 0, 0, 0)


    # 材料定义
    uniaxialMaterial('Steel01', 1, 345000000, 200000000000, 0.01)  # 柱材料
    uniaxialMaterial('Steel01', 2, 248000000, 200000000000, 0.01)  # 材料
    uniaxialMaterial('Elastic', 3, 1.0e20)


    # 纤维截面定义
    # 柱
    # W14*257
    sectionID = 11
    materialID = 1
    d = 0.4161
    bf = 0.4063
    tf = 0.048
    tw = 0.0298
    nfdw = 20
    nftw = 2
    nfbf = 20
    nftf = 2
    Wsection(sectionID, materialID, d, bf, tf, tw, nfdw, nftw, nfbf, nftf)

    # W14*283
    sectionID = 12
    materialID = 1
    d = 0.4252
    bf = 0.4092
    tf = 0.0526
    tw = 0.0328
    nfdw = 20
    nftw = 2
    nfbf = 20
    nftf = 2
    Wsection(sectionID, materialID, d, bf, tf, tw, nfdw, nftw, nfbf, nftf)
    # W14*370
    sectionID = 13
    materialID = 1
    d = 0.4552
    bf = 0.4185
    tf = 0.0676
    tw = 0.042
    nfdw = 20
    nftw = 2
    nfbf = 20
    nftf = 2
    Wsection(sectionID, materialID, d, bf, tf, tw, nfdw, nftw, nfbf, nftf)
    # W14*455
    sectionID = 14
    materialID = 1
    d = 0.4831
    bf = 0.4276
    tf = 0.0815
    tw = 0.0512
    nfdw = 20
    nftw = 2
    nfbf = 20
    nftf = 2
    Wsection(sectionID, materialID, d, bf, tf, tw, nfdw, nftw, nfbf, nftf)
    # W14*500
    sectionID = 15
    materialID = 1
    d = 0.4978
    bf = 0.4321
    tf = 0.0889
    tw = 0.0556
    nfdw = 20
    nftw = 2
    nfbf = 20
    nftf = 2
    Wsection(sectionID, materialID, d, bf, tf, tw, nfdw, nftw, nfbf, nftf)
    # 梁
    # W24*68
    sectionID = 21
    materialID = 2
    d = 0.6027
    bf = 0.2277
    tf = 0.0149
    tw = 0.0105
    nfdw = 20
    nftw = 2
    nfbf = 20
    nftf = 2
    Wsection(sectionID, materialID, d, bf, tf, tw, nfdw, nftw, nfbf, nftf)
    # W27*84
    sectionID = 22
    materialID = 2
    d = 0.6784
    bf = 0.253
    tf = 0.0163
    tw = 0.0117
    nfdw = 20
    nftw = 2
    nfbf = 20
    nftf = 2
    Wsection(sectionID, materialID, d, bf, tf, tw, nfdw, nftw, nfbf, nftf)

    # W30*99
    sectionID = 23
    materialID = 2
    d = 0.7531
    bf = 0.2654
    tf = 0.017
    tw = 0.0132
    nfdw = 20
    nftw = 2
    nfbf = 20
    nftf = 2
    Wsection(sectionID, materialID, d, bf, tf, tw, nfdw, nftw, nfbf, nftf)
    # W36*135
    sectionID = 24
    materialID = 2
    d = 0.903
    bf = 0.3035
    tf = 0.0201
    tw = 0.0152
    nfdw = 20
    nftw = 2
    nfbf = 20
    nftf = 2
    Wsection(sectionID, materialID, d, bf, tf, tw, nfdw, nftw, nfbf, nftf)
    # W36*160
    sectionID = 25
    materialID = 2
    d = 0.9147
    bf = 0.3048
    tf = 0.0259
    tw = 0.0165
    nfdw = 20
    nftw = 2
    nfbf = 20
    nftf = 2
    Wsection(sectionID, materialID, d, bf, tf, tw, nfdw, nftw, nfbf, nftf)

    # 建立截面
    section('Aggregator', 111, 3, 'T', '-section', 11)
    section('Aggregator', 112, 3, 'T', '-section', 12)
    section('Aggregator', 113, 3, 'T', '-section', 13)
    section('Aggregator', 114, 3, 'T', '-section', 14)
    section('Aggregator', 115, 3, 'T', '-section', 15)
    section('Aggregator', 121, 3, 'T', '-section', 21)
    section('Aggregator', 122, 3, 'T', '-section', 22)
    section('Aggregator', 123, 3, 'T', '-section', 23)
    section('Aggregator', 124, 3, 'T', '-section', 24)
    section('Aggregator', 125, 3, 'T', '-section', 25)

    # 定义构件
    geomTransf('Linear', 1)
    geomTransf('Linear', 2)

    # 柱
    # 1
    for i in range(1, 7):
        element('nonlinearBeamColumn', 10 + i, i, 10 + i, 5, 115, 1)

    # 1.5
    for i in range(1, 7):
        element('nonlinearBeamColumn', 20 + i, 10 + i, 100 + i, 5, 115, 1)

    # 2
    for i in range(1, 7):
        element('nonlinearBeamColumn', 30 + i, 100 + i, 20 + i, 5, 114, 1)

    # 3
    for i in range(1, 7):
        element('nonlinearBeamColumn', 40 + i, 20 + i, 30 + i, 5, 114, 1)

    # 3.5
    for i in range(1, 7):
        element('nonlinearBeamColumn', 50 + i, 30 + i, 200 + i, 5, 114, 1)

    # 4
    for i in range(1, 7):
        element('nonlinearBeamColumn', 60 + i, 200 + i, 40 + i, 5, 113, 1)

    # 5
    for i in range(1, 7):
        element('nonlinearBeamColumn', 70 + i, 40 + i, 50 + i, 5, 113, 1)

    # 5.5
    for i in range(1, 7):
        element('nonlinearBeamColumn', 80 + i, 50 + i, 300 + i, 5, 113, 1)

    # 6
    for i in range(1, 7):
        element('nonlinearBeamColumn', 90 + i, 300 + i, 60 + i, 5, 112, 1)

    # 7
    for i in range(1, 7):
        element('nonlinearBeamColumn', 100 + i, 60 + i, 70 + i, 5, 112, 1)

    # 7.5
    for i in range(1, 7):
        element('nonlinearBeamColumn', 110 + i, 70 + i, 400 + i, 5, 112, 1)

    # 8
    for i in range(1, 7):
        element('nonlinearBeamColumn', 120 + i, 400 + i, 80 + i, 5, 111, 1)

    # 9
    for i in range(1, 7):
        element('nonlinearBeamColumn', 130 + i, 80 + i, 90 + i, 5, 111, 1)


    # 梁
    # 1
    for i in range(1, 6):
        element('nonlinearBeamColumn', 210 + i, 10 + i, 10 + i + 1, 5, 125, 2)

    # 2
    for i in range(1, 6):
        element('nonlinearBeamColumn', 220 + i, 20 + i, 20 + i + 1, 5, 125, 2)

    # 3
    for i in range(1, 6):
        element('nonlinearBeamColumn', 230 + i, 30 + i, 30 + i + 1, 5, 124, 2)

    # 4
    for i in range(1, 6):
        element('nonlinearBeamColumn', 240 + i, 40 + i, 40 + i + 1, 5, 124, 2)

    # 5
    for i in range(1, 6):
        element('nonlinearBeamColumn', 250 + i, 50 + i, 50 + i + 1, 5, 124, 2)

    # 6
    for i in range(1, 6):
        element('nonlinearBeamColumn', 260 + i, 60 + i, 60 + i + 1, 5, 124, 2)

    # 7
    for i in range(1, 6):
        element('nonlinearBeamColumn', 270 + i, 70 + i, 70 + i + 1, 5, 123, 2)

    # 8
    for i in range(1, 6):
        element('nonlinearBeamColumn', 280 + i, 80 + i, 80 + i + 1, 5, 122, 2)

    # 9
    for i in range(1, 6):
        element('nonlinearBeamColumn', 290 + i, 90 + i, 90 + i + 1, 5, 121, 2)


    # 定义节点质量
    # 1层
    mass1 = 50465.75
    mass2 = 100931.5
    for i in range(1, 7):
        if i == 1 or i == 6:
            m = mass1
        else:
            m = mass2
        node_tag = 10 + i
        mass(node_tag, m, m, m)


    # 2-8层
    mass1 = 49416.46
    mass2 = 98832.93
    for j in range(2, 9):
        for i in range(1, 7):
            if i == 1 or i == 6:
                m = mass1
            else:
                m = mass2
            node_tag = j * 10 + i
            mass(node_tag, m, m, m)

    # 9层
    mass1 = 53463.72
    mass2 = 106927.4
    for i in range(1, 7):
        if i == 1 or i == 6:
            m = mass1
        else:
            m = mass2
        node_tag = 90 + i
        mass(node_tag, m, m, m)


    wipeAnalysis()
    # 特征值分析
    num_eigen = 10
    eigen_values = eigen(num_eigen)
    w2 = eigen_values[0]
    T_1 = 2 * np.pi / np.sqrt(w2)

    # 定义瑞利阻尼
    xDamp = 0.04  # 钢结构的阻尼比
    MpropSwitch = 1.0
    KcurrSwitch = 0.0
    KcommSwitch = 1.0
    KinitSwitch = 0.0
    omega1 = np.sqrt(eigen_values[0])
    omega3 = np.sqrt(eigen_values[2])
    alphaM = MpropSwitch * xDamp * (2 * omega1 * omega3) / (omega1 + omega3)
    betaKcurr = KcurrSwitch * 2. * xDamp / (omega1 + omega3)
    betaKcomm = KcommSwitch * 2. * xDamp / (omega1 + omega3)
    betaKinit = KinitSwitch * 2. * xDamp / (omega1 + omega3)
    rayleigh(alphaM, betaKcurr, betaKinit, betaKcomm)

    # 动力学分析
    load_tag_dynamic = 1
    pattern_tag_dynamic = 1
    X = 1

    timeSeries('Path', load_tag_dynamic, '-filePath', wave_file, '-dt', dt, '-factor', PGA)
    pattern('UniformExcitation', pattern_tag_dynamic, X, '-accel', load_tag_dynamic)

    analysis_dt = 0.01
    final_time = (NPTS - 1) * dt

    # NEWMARK直接积分
    wipeAnalysis()
    constraints('Plain')  # 约束条件
    numberer('RCM')  # 对节点重新编号
    system('BandGeneral')  # 解矩阵的方法
    test('EnergyIncr', 1.000000e-6, 10, 0, 2)  # 误差分析
    algorithm('KrylovNewton')  # 算法
    integrator('Newmark', 0.5, 0.25)  # 用newmark法积分
    analysis('Transient')  # 时程分析

    # 取时程数据的节点
    mark_nodes = [11, 21, 31, 41, 51, 61, 71, 81, 91]
    mark_nodes_len = len(mark_nodes)
    time_num = int(final_time / analysis_dt) + 100
    time = np.zeros(time_num)
    acceleration = np.zeros([mark_nodes_len, time_num])
    drift = np.zeros([mark_nodes_len, time_num])
    ok = 0
    time_idx = 0

    while ok == 0 and getTime() < final_time:
        current_time = getTime()
        ok = analyze(1, analysis_dt)
        for layer_idx in range(mark_nodes_len):
            acceleration[layer_idx, time_idx] = nodeAccel(mark_nodes[layer_idx], 1)
            drift[layer_idx, time_idx] = drift_cal(mark_nodes[layer_idx], height(mark_nodes[layer_idx]))
        time[time_idx] = current_time
        time_idx += 1

    a_max, a_avg = cal_a(acceleration)
    drift_max, drift_avg = cal_drift(drift)
    return [T_1, a_max, a_avg, drift_max, drift_avg]


def fitness(wave_features, original_model_resp, Rfc, model_a_max, model_a_avg, model_theta_max, model_theta_avg):
    """
    适应度函数
    :param wave_features: 地震波特征
    :param original_model_resp: 原结构地震响应
    :param Rfc: 分类模型
    :param model_a_max:
    :param model_a_avg:
    :param model_theta_max:
    :param model_theta_avg:
    :return:
    """
    def fitness_ga(p):
        [miu1, miu2, miu3, miu4, miu5, miu6, miu7, miu8, miu9,
        xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9,
        kappa1, kappa2, kappa3, kappa4, kappa5, kappa6, kappa7, kappa8, kappa9] = p

        miu = [miu1, miu2, miu3, miu4, miu5, miu6, miu7, miu8, miu9]
        xi = [xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9]
        kappa = kappa1, kappa2, kappa3, kappa4, kappa5, kappa6, kappa7, kappa8, kappa9
        md = np.zeros(9)
        cd = np.zeros(9)
        kd = np.zeros(9)
        # 惯容系统参数*
        # 1
        md[0] = miu[0] * 504675.54
        cd[0] = 2 * np.power(504675.54 * 17701517.6367, 0.5) * xi[0]
        kd[0] = 17701517.6367 * kappa[0]
        # 2
        md[1] = miu[1] * 494164.64
        cd[1] = 2 * np.power(494164.64 * 212966222.7839, 0.5) * xi[1]
        kd[1] = 212966222.7839 * kappa[1]
        # 3
        md[2] = miu[2] * 494164.64
        cd[2] = 2 * np.power(494164.64 * 198709990.1556, 0.5) * xi[2]
        kd[2] = 198709990.1556 * kappa[2]
        # 4
        md[3] = miu[3] * 494164.64
        cd[3] = 2 * np.power(494164.64 * 178923770.7303, 0.5) * xi[3]
        kd[3] = 178923770.7303 * kappa[3]
        # 5
        md[4] = miu[4] * 494164.64
        cd[4] = 2 * np.power(494164.64 * 168236912.4272, 0.5) * xi[4]
        kd[4] = 168236912.4272 * kappa[4]
        # 6
        md[5] = miu[5] * 494164.64
        cd[5] = 2 * np.power(494164.64 * 153412925.7595, 0.5) * xi[5]
        kd[5] = 153412925.7595 * kappa[5]
        # 7
        md[6] = miu[6] * 494164.64
        cd[6] = 2 * np.power(494164.64 * 117202366.5283, 0.5) * xi[6]
        kd[6] = 117202366.5283 * kappa[6]
        # 8
        md[7] = miu[7] * 494164.64
        cd[7] = 2 * np.power(494164.64 * 84968874.1376, 0.5) * xi[7]
        kd[7] = 84968874.1376 * kappa[7]
        # 9
        md[8] = miu[8] * 534637.2
        cd[8] = 2 * np.power(534637.2 * 58205933.4045, 0.5) * xi[8]
        kd[8] = 58205933.4045 * kappa[8]


        input_params = pd.DataFrame(data=None,
                                    columns=["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "k1", "k2", "k3",
                                             "k4", "k5", "k6", "k7", "k8", "k9", "c1", "c2", "c3", "c4", "c5", "c6", "c7",
                                             "c8", "c9", "sa1", "sa2", "sa3", "sa4", "sa5", "sa6", "sa7", "sa8", "sa9",
                                             "sa10", "sa11", "sa12", "sa13", "sa14", "sa15", "sa16", "sa17", "sa18", "sa19",
                                             "sa20", "sa21", "sa22", "sa23", "sa24", "sa25", "sa26", "sa27", "sa28", "sa29",
                                             "sa30", "sv1", "sv2", "sv3", "sv4", "sv5", "sv6", "sv7", "sv8", "sv9", "sv10",
                                             "sv11", "sv12", "sv13", "sv14", "sv15", "sv16", "sv17", "sv18", "sv19", "sv20",
                                             "sv21", "sv22", "sv23", "sv24", "sv25", "sv26", "sv27", "sv28", "sv29", "sv30",
                                             "sd1", "sd2", "sd3", "sd4", "sd5", "sd6", "sd7", "sd8", "sd9", "sd10", "sd11",
                                             "sd12", "sd13", "sd14", "sd15", "sd16", "sd17", "sd18", "sd19", "sd20", "sd21",
                                             "sd22", "sd23", "sd24", "sd25", "sd26", "sd27", "sd28", "sd29", "sd30", "sa_max",
                                             "sa_avg", "sv_max", "sv_avg", "sd_max", "sd_avg", "pga", "pgv", "pgd","epa",
                                             "epv", "epd", "pa", "pv", "pd", "ic"])

        a_max_ratio = 0
        a_avg_ratio = 0
        drift_max_ratio = 0
        drift_avg_ratio = 0
        for wave_name in original_model_resp.keys():
            [SA, SV, SD, SA_avg, SV_avg, SD_avg, SA_max, SV_max, SD_max, PGA, PGV,
             PGD, EPA, EPV, EPD, Pa, Pv, Pd,Ic] = wave_features[wave_name]
            res = []
            for i in md:
                res.append(i)
            for i in kd:
                res.append(i)
            for i in cd:
                res.append(i)
            for i in SA:
                res.append(i)
            for i in SV:
                res.append(i)
            for i in SD:
                res.append(i)
            res.append(SA_max)
            res.append(SA_avg)
            res.append(SV_max)
            res.append(SV_avg)
            res.append(SD_max)
            res.append(SD_avg)
            res.append(PGA)
            res.append(PGV)
            res.append(PGD)
            res.append(EPA)
            res.append(EPV)
            res.append(EPD)
            res.append(Pa)
            res.append(Pv)
            res.append(Pd)
            res.append(Ic)
            input_params.loc[len(input_params)] = res
            # 判断是否超限
            input_val_ok = input_params.loc[:, main_features_ok].values
            if not Rfc.predict(input_val_ok)[0]:
                return 1000
            origin_resp_tmp = original_model_resp[wave_name]
            a_max_origin = origin_resp_tmp[1]
            a_avg_origin = origin_resp_tmp[2]
            drift_max_origin = origin_resp_tmp[3]
            drift_avg_origin = origin_resp_tmp[4]
            # 预测结果
            input_val_a_max = input_params.loc[:, main_features_a_max].values
            input_val_a_avg = input_params.loc[:, main_features_a_avg].values
            input_val_drift_max = input_params.loc[:, main_features_drift_max].values
            input_val_drift_avg = input_params.loc[:, main_features_drift_avg].values

            a_max_predict = model_a_max.predict(input_val_a_max)[0]
            a_avg_predict = model_a_avg.predict(input_val_a_avg)[0]
            drift_max_predict = model_theta_max.predict(input_val_drift_max)[0]
            drift_avg_predict = model_theta_avg.predict(input_val_drift_avg)[0]
            a_max_ratio += a_max_predict / a_max_origin
            a_avg_ratio += a_avg_predict / a_avg_origin
            drift_max_ratio += drift_max_predict / drift_max_origin
            drift_avg_ratio += drift_avg_predict / drift_avg_origin
            # 清楚结果
            input_params = input_params.drop(index=0)

        wave_num = len(wave_features)
        a_max_ratio /= wave_num
        a_avg_ratio /= wave_num
        drift_max_ratio /= wave_num
        drift_avg_ratio /= wave_num
        fit_val = 1 / 4 * (a_max_ratio + a_avg_ratio + drift_max_ratio + drift_avg_ratio)
        return fit_val

    return fitness_ga

if __name__ == "__main__":

    # 导入训练好的模型
    Rfc = joblib.load("./ml_models/rfc.model")
    model_a_avg = joblib.load("./ml_models/rfr_a_avg.model")
    model_a_max = joblib.load("./ml_models/rfr_a_max.model")
    model_theta_avg = joblib.load("./ml_models/rfr_theta_avg.model")
    model_theta_max = joblib.load("./ml_models/rfr_theta_max.model")

    design_wave_names = ["Artificial_EQSignal1-Acc", "Artificial_EQSignal2-Acc", "RSN15_KERN_TAF021",
                         "RSN40_BORREGO_A-SON033", "RSN55_SFERN_BVP090", "RSN92_SFERN_WRP090", "RSN93_SFERN_WND143"]
    design_wave_dt = [0.02, 0.02, 0.01, 0.005, 0.005, 0.005, 0.005]
    design_wave_NPTS = [2000, 2000, 5435, 9045, 5330, 5955, 8000]
    design_wave_dir = "./waves_design_opensees/"
    original_model_resp = {}
    wave_features = {}
    for i in range(len(design_wave_names)):
        wave_file = design_wave_dir + design_wave_names[i] + '.txt'
        # 计算原结构响应
        original_model_resp[design_wave_names[i]] = original_model(wave_file, design_wave_NPTS[i], design_wave_dt[i], 1)
        # 计算地震波特征
        with open(wave_file, 'r') as fo:
            wave = np.loadtxt(wave_file)
        # 先这么设置
        xi = 0.02
        wave_features[design_wave_names[i]] = get_all_features(wave, design_wave_dt[i], xi)
    start = time.time()
    func = fitness(wave_features, original_model_resp, Rfc, model_a_max, model_a_avg, model_theta_max, model_theta_avg)
    ga = AGA(func=func, n_dim=27, size_pop=10, max_iter=2, lb=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ub=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], precision=1e-8,
             early_stop=500)
    best_x, best_y = ga.run()
    print('best_x:', best_x, '\n', 'best_y:', best_y)
    end = time.time()
    print('计算时间', end - start, 's')
    import pandas as pd
    import matplotlib.pyplot as plt

    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    plt.show()



