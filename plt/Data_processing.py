import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family']='Times New Roman'
plt.rcParams.update({'font.size': 20})
def moving_average(data, window_size):
    weights = np.repeat(1.0, window_size) / window_size
    smoothed_data = np.convolve(data, weights, 'valid')
    return smoothed_data

n_episode = 1000
window_size1 = 1
window_size2 = 1

#reward_maddpg_random = moving_average(np.load('3-Random_Reward_MADDPG_1000.npy'), window_size1)
reward_maddpg_bcd = moving_average(np.load('3-BCD_Reward_MADDPG_1000.npy'), window_size1)
#reward_maddpg_bcd1 = moving_average(np.load('3-Reward_MADDPG_1000_1.npy'), window_size1)
#reward_maddpg_bcd2 = moving_average(np.load('3-Reward_MADDPG_1000_2.npy'), window_size1)
#reward_maddpg_bcd3 = moving_average(np.load('3-Reward_MADDPG_1000_3.npy'), window_size1)
reward_ddpg = moving_average(np.load('3-Reward_DDPG_1000.npy'), window_size1)
#reward_sac = moving_average(np.load('3-Reward_SAC_1000.npy'), window_size1)

reward_maddpg0 = moving_average(np.load('3-BCD_User0_Reward_1000.npy'), window_size2)
reward_maddpg1 = moving_average(np.load('3-BCD_User1_Reward_1000.npy'), window_size2)
reward_maddpg2 = moving_average(np.load('3-BCD_User2_Reward_1000.npy'), window_size2)
reward_maddpg3 = moving_average(np.load('3-BCD_User3_Reward_1000.npy'), window_size2)
reward_maddpg4 = moving_average(np.load('3-BCD_User4_Reward_1000.npy'), window_size2)
reward_maddpg5 = moving_average(np.load('3-BCD_User5_Reward_1000.npy'), window_size2)
reward_maddpg6 = moving_average(np.load('3-BCD_User6_Reward_1000.npy'), window_size2)
reward_maddpg7 = moving_average(np.load('3-BCD_User7_Reward_1000.npy'), window_size2)

x1 =np.linspace(0,n_episode, n_episode, dtype=int)
x2 =np.linspace(0,n_episode, n_episode, dtype=int)

plt.figure(1, figsize=(8, 6.5))

plt.plot(x1, reward_ddpg, label='DDPG')
#plt.plot(x1, reward_maddpg_random, label='MADDPG-Random')
plt.plot(x1, reward_maddpg_bcd, label='Proposed algorithm')
#plt.plot(x1, reward_maddpg_bcd1, label='1')
#plt.plot(x1, reward_maddpg_bcd2, label='2')
#plt.plot(x1, reward_maddpg_bcd3, label='3')

#plt.plot(x1, reward_ddpg, label='DDPG')
#plt.plot(x1, reward_sac, label='SAC')
plt.grid(True, linestyle='-', linewidth=0.5)
# plt.yticks(y)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(loc='lower right', fontsize=20)
plt.savefig('D:\QKW\MARL-RIS-VEC\MARL-RIS-VEC\Reward.pdf', dpi=300, format='pdf')

plt.figure(2)
plt.subplot(2, 2, 1)
plt.plot(x2, reward_maddpg0)
plt.xlabel('Episode')
plt.ylabel('User0_Reward')

plt.subplot(2, 2, 2)
plt.plot(x2, reward_maddpg1)
plt.xlabel('Episode')
plt.ylabel('User1_Reward')

plt.subplot(2, 2, 3)
plt.plot(x2, reward_maddpg2)
plt.xlabel('Episode')
plt.ylabel('User2_Reward')

plt.subplot(2, 2, 4)
plt.plot(x2, reward_maddpg3)
plt.xlabel('Episode')
plt.ylabel('User3_Reward')

plt.tight_layout()

plt.figure(3)
plt.subplot(2, 2, 1)
plt.plot(x2, reward_maddpg4)
plt.xlabel('Episode')
plt.ylabel('User4_Reward')

plt.subplot(2, 2, 2)
plt.plot(x2, reward_maddpg5)
plt.xlabel('Episode')
plt.ylabel('User5_Reward')

plt.subplot(2, 2, 3)
plt.plot(x2, reward_maddpg6)
plt.xlabel('Episode')
plt.ylabel('User6_Reward')

plt.subplot(2, 2, 4)
plt.plot(x2, reward_maddpg7)
plt.xlabel('Episode')
plt.ylabel('User7_Reward')

plt.tight_layout()

plt.show()