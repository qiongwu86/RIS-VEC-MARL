import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rice

# 参数定义
fc = 28e9  # 载波频率 28 GHz
c = 3e8  # 光速
lambda_c = c / fc  # 波长
d = 100  # 通信距离 100 米
K_dB = 10  # 莱斯因子 (dB)
K = 10**(K_dB / 10)  # 转换为线性
num_samples = 10000  # 信号采样点数

# 路径损耗模型
def path_loss(distance, wavelength):
    return (wavelength / (4 * np.pi * distance))**2

# 莱斯信道生成
def rician_channel(K, num_samples):
    # LOS 分量
    LOS_component = np.sqrt(K / (K + 1))
    # NLOS 分量 (瑞利分布)
    NLOS_component = np.sqrt(1 / (2 * (K + 1))) * (
        np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
    )
    # 综合信道增益
    channel = LOS_component + NLOS_component
    return channel

# 生成莱斯信道
channel = rician_channel(K, num_samples)

# 路径损耗影响
path_loss_gain = np.sqrt(path_loss(d, lambda_c))
channel_with_path_loss = channel * path_loss_gain

# 绘制信道增益幅度分布
plt.figure(figsize=(8, 6))
plt.hist(np.abs(channel_with_path_loss), bins=100, density=True, alpha=0.7, label="莱斯信道增益幅度")
x = np.linspace(0, np.max(np.abs(channel_with_path_loss)), 1000)
pdf = rice.pdf(x, np.sqrt(K), scale=path_loss_gain / np.sqrt(2))
plt.plot(x, pdf, label="理论莱斯分布", color='r')
plt.title("莱斯信道增益幅度分布")
plt.xlabel("幅度")
plt.ylabel("概率密度")
plt.legend()
plt.grid()
plt.show()
