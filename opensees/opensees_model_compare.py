import random
from openseespy.opensees import *
import numpy as np
import matplotlib.pyplot as plt
import platform
import matplotlib
import os
import pandas as pd



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
    a_each_max = np.max(np.abs(a), 1)
    return [np.max(a_each_max), np.sum(a_each_max) / np.size(a_each_max)]


def cal_drift(drift):
    """
    计算层间位移角指标
    :param drift: 结构各层的层间位移角时程
    :return: 各层中最大的最大位移角和平均的最大位移角
    """
    drift_each_max = np.max(np.abs(drift), 1)
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
    return [T_1, a_max, a_avg, drift_max, drift_avg, time, acceleration, drift]


def TVMD_model(md, kd, cd, wave_file, NPTS, dt, PGA):
    """
    构建非线性模型，计算目标参数
    :param md: 每层TVMD的惯容系数
    :param kd: 每层TVMD的刚度
    :param cd: 每层TVMD的阻尼系数
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

    # 惯容系统参数*
    # 1
    inerterDamper2DX(3, 14, cd[0], md[0], kd[0])
    # 2
    inerterDamper2DX(13, 24, cd[1], md[1], kd[1])
    # 3
    inerterDamper2DX(23, 34, cd[2], md[2], kd[2])
    # 4
    inerterDamper2DX(33, 44, cd[3], md[3], kd[3])
    # 5
    inerterDamper2DX(43, 54, cd[4], md[4], kd[4])
    # 6
    inerterDamper2DX(53, 64, cd[5], md[5], kd[5])
    # 7
    inerterDamper2DX(63, 74, cd[6], md[6], kd[6])
    # 8
    inerterDamper2DX(73, 84, cd[7], md[7], kd[7])
    # 9
    inerterDamper2DX(83, 94, cd[8], md[8], kd[8])

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
    return [T_1, a_max, a_avg, drift_max, drift_avg, time, acceleration, drift]


def compare(wave_name, NPTS, dt, PGA, miu, xi, kappa):
    """
    随机生成TVMD结构，并随机选择一条波进行计算，返回计算结果
    :param miu: 惯质比
    :param xi: 阻尼比
    :param kappa: 刚度比
    :param wave_names: 候选波的集合
    :param NPTS: 数据点数
    :param dt: 时间间隔
    :param PGA: 地震波放大系数
    :return: 设计参数和计算结果
    """
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

    wave_file = "./waves_design_opensees/" + wave_name + ".txt"

    cal_res_tvmd = TVMD_model(md, kd, cd, wave_file, NPTS, dt, PGA)
    cal_res_origin = original_model(wave_file, NPTS, dt, PGA)

    res = [cal_res_tvmd, cal_res_origin]

    return res


def draw_acc_time_history(res, wave_index, layer_index):
    """
    绘制加速度时程
    :param res: 对比计算结果
    :param wave_index: 需要绘制的地震波索引
    :param layer_index: 需要绘制的层的层号
    """
    acc_res_index = 6
    time_res_index = 5
    # 通过分析时间计算绘图的索引范围
    analysis_dt = 0.01
    time = np.array(res[wave_index][0][time_res_index])
    index_range = 0
    for i in range(len(time) - 1, 0, -1):
        if time[i] != 0.0:
            index_range = i
            break
    time = time[0: index_range]
    # 绘制加速度时程
    # TVMD的加速度
    a_tvmd = np.array(res[wave_index][0][acc_res_index][layer_index][0: index_range])
    # 原结构加速度
    a_origin = np.array(res[wave_index][1][acc_res_index][layer_index][0: index_range])

    plt.plot(time, a_origin, 'b', label="原结构")
    plt.plot(time, a_tvmd, 'r', label="TVMD减震结构")
    plt.xlim(0, analysis_dt * time.size)
    plt.xlabel("时间 (s)")
    plt.ylabel("加速度 (s)")
    plt.legend()
    plt.show()

def draw_drift(res):
    """
    绘制平均层间位移角
    :param res: 对比计算结果
    """
    drift_res_index = 7
    layer = np.arange(1, 10, 1)
    drift_origin = np.zeros(9)
    drift_tvmd = np.zeros(9)
    wave_num = len(res)
    # 绘制层间位移角均值
    for i in range(wave_num):
        drift_tvmd = drift_tvmd + np.max(np.abs(res[i][0][drift_res_index]), 1)
        drift_origin = drift_origin + np.max(np.abs(res[i][1][drift_res_index]), 1)
    drift_tvmd = drift_tvmd / wave_num * 100
    drift_origin = drift_origin / wave_num * 100
    plt.plot(drift_origin, layer, 'b', label="原结构")
    plt.plot(drift_tvmd, layer, 'r', label="TVMD减震结构")
    plt.xlabel("层间位移角 (%)")
    plt.ylabel("层号")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 读取全部的地震波名，并读取每个波的NPTS和dt
    design_wave_names = ["Artificial_EQSignal1-Acc", "Artificial_EQSignal2-Acc", "RSN15_KERN_TAF021",
                         "RSN40_BORREGO_A-SON033", "RSN55_SFERN_BVP090", "RSN92_SFERN_WRP090", "RSN93_SFERN_WND143"]
    design_wave_dt = [0.02, 0.02, 0.01, 0.005, 0.005, 0.005, 0.005]
    design_wave_NPTS = [2000, 2000, 5435, 9045, 5330, 5955, 8000]

    miu = [0.94399752, 0.87049454, 0.31790556, 0.07476673, 0.45662818, 0.98182482, 0.27824921, 0.26211521, 0.55353305]
    xi = [0.41125741, 0.39079743, 0.4275923, 0.90192352, 0.77373728, 0.40159259, 0.54819759, 0.85856897, 0.25299614]
    kappa = [0.00161348, 0.31563436, 0.74299302, 0.98007412, 0.60877773, 0.6290415, 0.82586106, 0.99978249, 0.39623232]

    res = []
    for i in range(7):
        res.append(compare(design_wave_names[i], design_wave_NPTS[i], design_wave_dt[i], 1, miu, xi, kappa))

    fitness = 0
    for i in range(7):
        res_tvmd = res[i][0]
        res_origin = res[i][1]
        for j in range(1, 5):
            fitness += res_tvmd[j] / res_origin[j]
    print(fitness / 7)

    # 绘图
    # 解决中文显示问题
    sys_platform = platform.platform().lower()
    if "windows" in sys_platform:
        font = {
            "family": "Microsoft YaHei"
        }
        matplotlib.rc("font", **font)
    else:
        font = {
            "family": "Arial Unicode MS"
        }
        matplotlib.rc("font", **font)
    draw_acc_time_history(res, 0, 0)
    draw_drift(res)
# %%




