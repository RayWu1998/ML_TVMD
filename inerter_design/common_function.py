import numpy as np
from numba import jit


@jit(nopython=True)
def newmark(K, M, C, ag, dt, E):
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

    for i in range(0, length - 1):
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
        ddy_ab[..., i] = ddy[..., i] + ag[i]
    return y, dy, ddy, ddy_ab


@jit(nopython=True)
def newmark_one(k, m, c, ag, dt):
    """
        利用Newmarkβ计算结构的时程响应，单自由度
        :param k: 刚度
        :param m: 质量
        :param c: 阻尼
        :param ag: 加速度时程
        :param dt: 时间间隔
        :return: 结构的时程信息
        """
    length = len(ag)
    y = np.zeros(length, dtype=float)
    dy = np.zeros(length, dtype=float)
    ddy = np.zeros(length, dtype=float)
    ddy_ab = np.zeros(length, dtype=float)

    for i in range(0, length - 1):
        z_acc = ag[i + 1] - ag[i]
        z_y = (-m * z_acc + (4 / dt) * m * dy[i] + 2 * m * ddy[i] + 2 * c * dy[i]) / (
                    k + 2 * c / dt + m * 4 / (dt ** 2))
        y[i + 1] = y[i] + z_y
        z_dy = 2 / dt * z_y - 2 * dy[i]
        dy[i + 1] = dy[i] + z_dy
        z_ddy = 4 / (dt ** 2) * z_y - (4 / dt) * dy[i] - 2 * ddy[i]
        ddy[i + 1] = ddy[i] + z_ddy
        ddy_ab[i] = ddy[i] + ag[i]
    return y, dy, ddy, ddy_ab


def eigenvalue(M, K):
    """
    通过求解K和M的广义特征值和特征向量，获得结构各阶的自振频率，自振周期，模态
    :param M:
    :param K:
    :return:
            w: 自振频率
            T: 自振周期
            phi: 模态
            gamma: 振型参与系数
            M_par: 质量参与系数
    """
    dof = len(M)
    E = np.ones([dof, 1])
    m_sum = np.sum(M)
    A = np.dot(np.linalg.inv(M), K)
    omega, phi = np.linalg.eig(A)
    w = np.sqrt(omega)
    T = 2 * np.pi / w
    gamma = np.zeros(dof)
    M_par = np.zeros(dof)
    for i in range(dof):
        phi_i = phi[:, i, None]
        M_gen = np.dot(np.dot(phi_i.T, M), phi_i).reshape(1)
        gamma[i] = np.dot(np.dot(phi[i, :, None].T, M), E) / M_gen
        M_par[i] = M_gen * gamma[i] ** 2 / m_sum
        phi[:, i, None] = phi[:, i, None] / np.max(np.abs(phi[:, i, None]))
    # 重新排序
    indices = np.argsort(w)
    w = w[indices]
    T = T[indices]
    gamma = gamma[indices]
    M_par = M_par[indices]
    phi = phi[:, indices]
    return w, T, phi, gamma, M_par







