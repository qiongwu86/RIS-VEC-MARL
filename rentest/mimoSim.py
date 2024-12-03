import numpy as np
import matplotlib.pyplot as plt

# 参数定义
fc = 28e9  # 载波频率 28 GHz
c = 3e8  # 光速
lambda_c = c/fc  # 波长
d = 100  # 通信距离 100 米
K_dB = 10  # 莱斯因子 (dB)
K = 10 ** (K_dB/10)  # 转换为线性
N_t = 4  # 发射天线数
N_r = 4  # 接收天线数
num_samples = 1000  # 信号采样点数

# 配置定义
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 路径损耗模型
def path_loss(distance, wavelength):
    return (wavelength/(4*np.pi*distance)) ** 2


# 生成莱斯信道的单个元素
def rician_channel_element(K, num_samples=1):
    LOS_component = np.sqrt(K/(K + 1))
    NLOS_component = np.sqrt(1/(2*(K + 1)))*(
            np.random.randn(num_samples) + 1j*np.random.randn(num_samples)
    )
    channel = LOS_component + NLOS_component
    return channel


# 生成 MIMO 莱斯信道
def rician_mimo_channel(K, N_r, N_t):
    H = np.zeros((N_r, N_t), dtype=complex)
    for i in range(N_r):
        for j in range(N_t):
            H[i, j] = rician_channel_element(K)
    return H


# 路径损耗影响
path_loss_gain = np.sqrt(path_loss(d, lambda_c))

# 生成多次样本
mimo_channels = []
for _ in range(num_samples):
    H = rician_mimo_channel(K, N_r, N_t)
    mimo_channels.append(H*path_loss_gain)


# 波束成形权重
def beamforming_weights(H):
    # 使用信道的主奇异向量作为权重
    U, S, Vh = np.linalg.svd(H)
    w_r = U[:, 0]  # 接收波束成形权重
    w_t = Vh[0, :].conj()  # 发射波束成形权重
    return w_r, w_t


# 波束成形后的信道增益
def beamformed_channel_gain(H, w_r, w_t):
    return np.abs(np.dot(w_r.conj().T, np.dot(H, w_t)))


# 比较波束成形前后的信道增益
raw_gains = []
beamformed_gains = []

for H in mimo_channels:
    w_r, w_t = beamforming_weights(H)
    raw_gain = np.linalg.norm(H)  # 波束成形前的信道增益
    beamformed_gain = beamformed_channel_gain(H, w_r, w_t)  # 波束成形后的信道增益
    raw_gains.append(raw_gain)
    beamformed_gains.append(beamformed_gain)

# 可视化增益分布
plt.figure(figsize=(12, 6))

plt.hist(raw_gains, bins=50, alpha=0.7, label="波束成形前")
plt.hist(beamformed_gains, bins=50, alpha=0.7, label="波束成形后")

plt.title("信道增益分布：波束成形前后对比")
plt.xlabel("信道增益")
plt.ylabel("概率密度")
plt.legend()
plt.grid()
plt.show()
