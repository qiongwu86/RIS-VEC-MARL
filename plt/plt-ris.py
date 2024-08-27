import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})

x = [20, 40, 60, 80, 100, 120]

x3 = [2.88, 2.44, 2.24, 2.15, 2.08, 2]
y3 = [41.9, 35.56, 32.76, 31.4, 30.45, 29.8]

x2 = [2.61, 2.19, 2.1, 2.04, 1.997, 1.96]
y2 = [37.8, 32.1, 30.67, 29.83, 29.21, 28.81]

x1 = [2, 1.783867, 1.75, 1.7, 1.66, 1.64]
y1 = [29.567, 26.32, 25.94, 25.47, 25.12, 24.92]

# 创建图像和子图
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(x, x1, 'o-', markersize=5, linestyle='solid', label='Proposed Algorithm')
ax1.plot(x, x3, '^-', markersize=5, linestyle='solid', label='MADDPG Random')
ax1.plot(x, x2, '+-', markersize=5, linestyle='solid', color='skyblue', label='MADDPG SA')

ax1.set_ylabel('Total power consumption')  # 添加y轴标签

ax2.plot(x, y1, 'o-', markersize=5, linestyle='solid', label='Proposed Algorithm')
ax2.plot(x, y3, '^-', markersize=5, linestyle='solid', label='MADDPG Random')
ax2.plot(x, y2, '+-', markersize=5, linestyle='solid', color='skyblue', label='MADDPG SA')

ax2.set_ylabel('Buffer length', fontsize=14)  # 添加y轴标签
ax2.set_xlabel('The number of RIS elements', fontsize=14)

# 调整子图位置
plt.subplots_adjust(top=0.95, bottom=0.15)

ax1.grid(True)
ax2.grid(True)

# 添加图例
ax1.legend()
ax2.legend()

plt.savefig('D:\QKW\MARL-RIS-VEC\qkw-letter-final\MARL-RIS-VEC\RIS.pdf', dpi=300, format='pdf')
plt.show()
