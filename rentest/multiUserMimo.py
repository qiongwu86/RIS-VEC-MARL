import numpy as np
import matplotlib.pyplot as plt
import numpy.random

numpy.random.seed(114514)
# 参数定义
fc = 28e9  # 载波频率 28 GHz
c = 3e8  # 光速
lambda_c = c/fc  # 波长
d = 100  # 通信距离 100 米
K_dB = 10  # 莱斯因子 (dB)
K = 10 ** (K_dB/10)  # 转换为线性
N_t = 8  # 发射天线数
N_r = 4  # 接收天线数
num_users = 4  # 用户数量
noise_power = 1e-9  # 噪声功率
power_constraint = 1  # 假设总功率约束为 1 瓦

# 配置定义
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 路径损耗模型
def path_loss(distance, wavelength):
    return (wavelength/(4*np.pi*distance)) ** 2


# 生成 MIMO 莱斯信道
def rician_mimo_channels(num_users, N_r, N_t):
    # 路径损耗增益
    path_loss_gain = np.sqrt(path_loss(d, lambda_c))

    # 批量生成 LOS 和 NLOS 分量
    LOS_component = np.sqrt(K/(K + 1))  # LOS 比例因子
    NLOS_component = np.sqrt(1/(K + 1))  # NLOS 比例因子

    # 生成所有用户的信道矩阵
    # 形状为 (num_users, N_r, N_t)
    LOS_matrix = LOS_component*(np.ones((num_users, N_r, N_t), dtype=complex))
    NLOS_matrix = NLOS_component*(np.random.randn(num_users, N_r, N_t) +
                                  1j*np.random.randn(num_users, N_r, N_t))

    # 总信道矩阵（包含路径损耗）
    channels = (LOS_matrix + NLOS_matrix)*path_loss_gain
    return channels


def mmse_beamforming(channels):
    # 计算 MMSE 波束成形矩阵
    H = np.vstack(channels)
    HH_hermitian = H.conj().T @ H
    W_mmse = np.linalg.pinv(HH_hermitian + noise_power * np.eye(H.shape[1])) @ H.conj().T
    return W_mmse


# 零强迫波束成形
def zero_forcing_beamforming(channels):
    H_combined = np.vstack(channels)  # 合并所有用户的信道矩阵
    W_t = np.linalg.pinv(H_combined)  # 计算伪逆矩阵
    return W_t


def calculate_sinr(H_k, W, k, num_users, N_r):
    """
    计算第 k 个用户的每根接收天线的 SINR。
    """
    sinrs = []
    W_k = W[:, k * N_r:(k + 1) * N_r]  # 获取用户 k 的波束成形矩阵部分
    for r in range(N_r):
        h_rk = H_k[r, :]  # 第 r 根接收天线的信道向量
        # 信号功率
        signal_power = np.abs(np.dot(h_rk, W_k[:, r])) ** 2
        # 干扰功率（排除用户 k 的波束成形部分）
        interference_power = 0
        for j in range(num_users):
            if j != k:
                W_j = W[:, j * N_r:(j + 1) * N_r]  # 获取用户 j 的波束成形矩阵部分
                interference_power += np.sum(np.abs(np.dot(h_rk, W_j)) ** 2)
        # SINR 计算
        sinr_rk = signal_power / (interference_power + noise_power)
        sinrs.append(10 * np.log10(sinr_rk))  # 转为 dB
    return np.mean(sinrs)  # 返回平均 SINR


def normalize_power(W, power_constraint):
    """
    功率归一化函数，确保波束成形矩阵的总功率小于或等于给定的功率约束。
    W: 波束成形矩阵
    power_constraint: 发射功率约束（标量）
    """
    # 计算原始波束成形矩阵的功率
    current_power = np.sum(np.abs(W) ** 2)

    # 归一化波束成形矩阵
    if current_power > power_constraint:
        normalization_factor = np.sqrt(power_constraint/current_power)
        W = W*normalization_factor
    return W


# 仿真多用户波束成形
channels = rician_mimo_channels(num_users, N_r, N_t)

W_mmse = mmse_beamforming(channels)
W_zf = zero_forcing_beamforming(channels)
W_mmse = normalize_power(W_mmse, power_constraint)
W_zf = normalize_power(W_zf, power_constraint)

# 计算每个用户的 SINR
sinrs_mmse = []
sinrs_zf = []

for k in range(num_users):
    H_k = channels[k]  # 用户 k 的信道矩阵
    # 计算每个用户的 SINR
    sinr_mmse = calculate_sinr(H_k, W_mmse, k, num_users, N_r)
    sinr_zf = calculate_sinr(H_k, W_zf, k, num_users, N_r)

    sinrs_mmse.append(sinr_mmse)
    sinrs_zf.append(sinr_zf)

plt.figure(figsize=(10, 6))
plt.bar(range(1, num_users + 1), sinrs_mmse, alpha=0.7, label='MMSE SINR')
plt.bar(range(1, num_users + 1), sinrs_zf, alpha=0.7, label='ZF SINR')
plt.title("MMSE 与 ZF 波束成形的 SINR 对比")
plt.xlabel("用户编号")
plt.ylabel("SINR (dB)")
plt.legend()
plt.grid()
plt.show()
