import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})
# 数据
x = np.arange(1, 4.5, 0.5)
y1 = [9.24, 13.893, 18.8, 23.5, 29, 36.21, 44.3]
y2 = [11.92, 19.1, 27.956, 38.72, 55.24, 76.667, 97.74]
y3 = [8.21, 13.1367, 27.732, 96.93, 100, 100, 100]#389, 590, 1001]
y4 = [8.75, 13.137, 17.7, 21.83, 26.32, 31.3348, 36.15]
y5 = [10.04, 15.68, 21.8, 28.09, 35.43, 44.92, 57]
y6 = [7.857, 12.01, 16.2366, 20.87, 26.32, 33.5127, 77.8]
y7 = [8.36, 12.566, 16.9, 20.87, 25, 29.46, 33.47]
y8 = [8.93, 13.56, 18.6, 23.22, 28.227, 33.57, 39]
y9 = [7.842, 11.965, 15.86652, 19.8365, 23.65, 27.885, 32.16]

# 画图
plt.figure()

plt.plot(x, y1, 'o--', markersize=5, label='Proposed Algorithm, c=1e-27')
plt.plot(x, y2, 'o--', markersize=5, label='MADDPG Random, c=1e-27')
plt.plot(x, y3, 'o-.', markersize=5, label='DDPG, c=1e-27 (Truncated at 100)')
plt.plot(x, y4, 's-', markersize=5, label='Proposed Algorithm, c=1e-28')
plt.plot(x, y5, 's-', markersize=5, label='MADDPG Random, c=1e-28')
plt.plot(x, y6, 's-', markersize=5, label='DDPG, c=1e-28')
plt.plot(x, y7, '^:', markersize=5, label='Proposed Algorithm, c=1e-29')
plt.plot(x, y8, '^:', markersize=5, label='MADDPG Random, c=1e-29')
plt.plot(x, y9, '^:', markersize=5, label='DDPG, c=1e-29')

plt.xlabel('Task arrival rate / Mbps')
plt.ylabel('Buffer length')
#plt.title('Line Plot of Different Algorithms')
plt.xticks(x, [str(val) for val in x])  # 将横坐标设置为字符串表示，确保显示为小数点后一位
plt.ylim(0, 110)  # 设置y轴范围

plt.legend(loc='upper left', fontsize=11)  # 图例位置
plt.grid(True)
plt.tight_layout()
plt.savefig('power-buffer.pdf', dpi=300, format='pdf')
plt.show()