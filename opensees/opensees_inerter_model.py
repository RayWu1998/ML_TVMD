from openseespy.opensees import *
import numpy as np
import matplotlib.pyplot as plt
import opsvis as opsv
import os

# 9层框架钢结构Benchmark模型的OpenSeesPy实现
print("=========================================================")
print("Start 2D Steel Frame Example")


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
    print("Wsection {} was established!".format(secID))


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
        return 5.4864 # 1层
    else:
        return 3.9624 # 2~9层


def TVMD_model():
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
            print("node {} {} {}".format(node_tag, x, y))

    print("Nodes have been defined")

    # 节点约束
    for i in range(0, 10):
        for j in range(1, 7):
            if i == 0:
                fix(i * 10 + j, 1, 1, 1)
            else:
                fix(i * 10 + j, 0, 0, 0)

    print("Nodes have been fixed")

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
        print("node {} {} {}".format(node_tag, x, y))

    for i in range(1, 7):
        node_tag = 200 + i
        x = span_len * (i - 1)
        y = bh2
        node(node_tag, x, y)
        fix(node_tag, 0, 0, 0)
        print("node {} {} {}".format(node_tag, x, y))

    for i in range(1, 7):
        node_tag = 300 + i
        x = span_len * (i - 1)
        y = bh3
        node(node_tag, x, y)
        fix(node_tag, 0, 0, 0)
        print("node {} {} {}".format(node_tag, x, y))

    for i in range(1, 7):
        node_tag = 400 + i
        x = span_len * (i - 1)
        y = bh4
        node(node_tag, x, y)
        fix(node_tag, 0, 0, 0)
        print("node {} {} {}".format(node_tag, x, y))

    print("Variable cross-section nodes have been defined")

    # 材料定义
    uniaxialMaterial('Steel01', 1, 345000000, 200000000000, 0.01)  # 柱材料
    uniaxialMaterial('Steel01', 2, 248000000, 200000000000, 0.01)  # 材料
    uniaxialMaterial('Elastic', 3, 1.0e20)
    print("Materials have been defined!")

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
    print("Fiber section have been defined!")

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

    print("Columns have been defined!")

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

    print("Beams have been defined!")

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
        print("mass {} {} {} {}".format(node_tag, m, m, m))
    print("Total mass of floor 1 is {}".format(2 * mass1 + 4 * mass2))

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
            print("mass {} {} {} {}".format(node_tag, m, m, m))
        print("Totalmass of floor {} is {}".format(j, 2 * mass1 + 4 * mass2))

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
        print("mass {} {} {} {}".format(node_tag, m, m, m))

    print("Total mass of floor 9 is {}".format(2 * mass1 + 4 * mass2))

    # source inerter_parameter.tcl;
    miu = 1.62993
    xi = 0.014488
    kappa = 0.01
    # 系数
    xishu1 = 0.6689
    xishu2 = 0.8764
    xishu3 = 0.9121
    xishu4 = 0.9441
    xishu5 = 0.8481
    xishu6 = 0.7302
    xishu7 = 0.8610
    xishu8 = 1
    xishu9 = 0.8299

    # 惯容系统参数*
    # 1
    md1 = xishu1 * miu * 504675.54
    cd1 = 2 * np.power(504675.54 * 17701517.6367, 0.5) * xishu1 * xi
    kd1 = 17701517.6367 * xishu1 * kappa
    inerterDamper2DX(3, 14, cd1, md1, kd1)
    # 2
    md2 = xishu2 * miu * 494164.64
    cd2 = 2 * np.power(494164.64 * 212966222.7839, 0.5) * xishu2 * xi
    kd2 = 212966222.7839 * xishu2 * kappa
    inerterDamper2DX(13, 24, cd2, md2, kd2)
    # 3
    md3 = xishu3 * miu * 494164.64
    cd3 = 2 * np.power(494164.64 * 198709990.1556, 0.5) * xishu3 * xi
    kd3 = 198709990.1556 * xishu3 * kappa
    inerterDamper2DX(23, 34, cd3, md3, kd3)
    # 4
    md4 = xishu4 * miu * 494164.64
    cd4 = 2 * np.power(494164.64 * 178923770.7303, 0.5) * xishu4 * xi
    kd4 = 178923770.7303 * xishu4 * kappa
    inerterDamper2DX(33, 44, cd4, md4, kd4)
    # 5
    md5 = xishu5 * miu * 494164.64
    cd5 = 2 * np.power(494164.64 * 168236912.4272, 0.5) * xishu5 * xi
    kd5 = 168236912.4272 * xishu5 * kappa
    inerterDamper2DX(43, 54, cd5, md5, kd5)
    # 6
    md6 = xishu6 * miu * 494164.64
    cd6 = 2 * np.power(494164.64 * 153412925.7595, 0.5) * xishu6 * xi
    kd6 = 153412925.7595 * xishu6 * kappa
    inerterDamper2DX(53, 64, cd6, md6, kd6)
    # 7
    md7 = xishu7 * miu * 494164.64
    cd7 = 2 * np.power(494164.64 * 117202366.5283, 0.5) * xishu7 * xi
    kd7 = 117202366.5283 * xishu7 * kappa
    inerterDamper2DX(63, 74, cd7, md7, kd7)
    # 8
    md8 = xishu8 * miu * 494164.64
    cd8 = 2 * np.power(494164.64 * 84968874.1376, 0.5) * xishu8 * xi
    kd8 = 84968874.1376 * xishu8 * kappa
    inerterDamper2DX(73, 84, cd8, md8, kd8)
    # 9
    md9 = xishu9 * miu * 534637.2
    cd9 = 2 * np.power(534637.2 * 58205933.4045, 0.5) * xishu9 * xi
    kd9 = 58205933.4045 * xishu9 * kappa
    inerterDamper2DX(83, 94, cd9, md9, kd9)

    wipeAnalysis()
    # 特征值分析
    num_eigen = 10
    eigen_values = eigen(num_eigen)
    for i in range(0, num_eigen):
        w2 = eigen_values[i]
        T = 2 * np.pi / np.sqrt(w2)
        print("natural period {}: {}".format(i, T))

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

    displacement = {}
    velocity = {}
    acceleration = {}
    drift = {}
    # 动力学分析
    PGA = 5.1
    dt = 0.02
    nPts = 1955
    timeSeries('Path', 2, '-filePath', 'AW2.txt', '-dt', dt, '-factor', PGA)
    pattern('UniformExcitation', 2, 1, '-accel', 2)

    for i in [11, 21, 31, 41, 51, 61, 71, 81, 91]:
        displacement['disp' + str(i)] = []
        velocity['vel' + str(i)] = []
        acceleration['accel' + str(i)] = []
        drift['drift' + str(i)] = []

    time = []
    final_time = 1955 * 0.02

    # NEWMARK直接积分
    wipeAnalysis()
    constraints('Plain')  # 约束条件
    numberer('RCM')  # 对节点重新编号
    system('BandGeneral')  # 解矩阵的方法
    test('EnergyIncr', 1.000000e-6, 10, 0, 2)  # 误差分析
    algorithm('KrylovNewton')  # 算法
    integrator('Newmark', 0.5, 0.25)  # 用newmark法积分
    analysis('Transient')  # 时程分析

    while getTime() < final_time:
        current_time = getTime()
        analyze(1, 0.02)

        for i in [11, 21, 31, 41, 51, 61, 71, 81, 91]:
            displacement['disp' + str(i)].append(nodeDisp(i, 1))
            velocity['vel' + str(i)].append(nodeVel(i, 1))
            acceleration['accel' + str(i)].append(nodeAccel(i, 1))

        time.append(current_time)

    plt.plot(time, displacement['disp11'])
    print(np.max(displacement['disp11']))
    print(np.min(displacement['disp11']))
    plt.show()

    plt.plot(time, velocity['vel11'])
    plt.show()

    plt.plot(time, acceleration['accel11'])
    plt.show()

    # 显示模型
    opsv.plot_model()
    plt.show()


if __name__ == "__main__":
    TVMD_model()
