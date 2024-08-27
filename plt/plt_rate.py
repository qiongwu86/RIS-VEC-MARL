import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置字体
rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})

# 数据
x = [1, 1.5, 2, 2.5, 3, 3.5, 4]
y1 = [1.11, 1.225, 1.3697, 1.55, 1.7838678, 2.08, 2.41]
y2 = [1.163, 1.346, 1.59, 1.95, 2.42, 3.11, 3.94]
y3 = [7.56, 8.067, 7.566, 7.573, 7.4783, 7.827, 7.76]
y4 = [1.127, 1.25968, 1.43, 1.66, 1.9858, 2.46, 3.01]
y5 = [1.228, 1.51, 1.962, 2.677, 3.88, 5.395, 6.95]
y6 = [7.56, 8.067, 7.566, 7.573, 7.4783, 7.827, 7.76]
y7 = [1.1, 1.2, 1.33, 1.48, 1.67, 1.92, 2.18]
y8 = [1.13, 1.26, 1.44, 1.6556, 1.93, 2.276, 2.582]
y9 = [7.56, 8.067, 7.566, 7.573, 7.4783, 7.827, 7.76]

# 绘图
plt.figure()

plt.plot(x, y4, 'o--', markersize=5, label='Proposed Algorithm, c=1e-27')
plt.plot(x, y5, 'o--', markersize=5, label='MADDPG Random, c=1e-27')
plt.plot(x, y6, 'o--', markersize=5, label='DDPG, c=1e-27')

plt.plot(x, y1, 's-', markersize=5, label='Proposed Algorithm, c=1e-28')
plt.plot(x, y2, 's-', markersize=5, label='MADDPG Random, c=1e-28')
plt.plot(x, y3, 's-', markersize=5, label='DDPG, c=1e-28')

plt.plot(x, y7, '^:', markersize=5, label='Proposed Algorithm, c=1e-29')
plt.plot(x, y8, '^:', markersize=5, label='MADDPG Random, c=1e-29')
plt.plot(x, y9, '^:', markersize=5, label='DDPG, c=1e-29')

plt.xticks(x)
plt.xlabel('Task arrival rate / Mbps')
plt.ylabel('Total power consumption')
plt.grid(True, linestyle='-', linewidth=0.5)
plt.legend(bbox_to_anchor=(0.5, 0.88), fontsize=12)

plt.tight_layout()
plt.savefig('power-rate.pdf', dpi=300, format='pdf')
plt.show()
