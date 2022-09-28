
from tkinter import font
import matplotlib.pyplot as plt
import matplotlib.font_manager
import pandas as pd

SMOOTH_FACTOR = 0.6
ALPHA = 0.3
EPS = 150

df_sac_sr = pd.read_csv('./logs/SAC/evaluate/SR.csv') # Path of csv file
df_sac_cr = pd.read_csv('./logs/SAC/evaluate/CR.csv') # Path of csv file
df_sac_sr_smoothed = df_sac_sr.ewm(alpha=(1 - SMOOTH_FACTOR)).mean()
df_sac_cr_smoothed = df_sac_cr.ewm(alpha=(1 - SMOOTH_FACTOR)).mean()

sac_sr_eps = df_sac_sr['Step'].values.tolist()
sac_sr = df_sac_sr['Value'].values.tolist()
sac_sr_smoothed = df_sac_sr_smoothed['Value'].values.tolist()
sac_cr_eps = df_sac_cr['Step'].values.tolist()
sac_cr = df_sac_cr['Value'].values.tolist()
sac_cr_smoothed = df_sac_cr_smoothed['Value'].values.tolist()

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
# plt.figure()
# p = plt.plot(sac_sr_eps, sac_sr, label='raw', alpha=ALPHA)
# plt.plot(sac_sr_eps, sac_sr_smoothed, label='smoothed, α=0.6', color=p[0].get_color())
# plt.xlabel('agents', fontsize=16)
# plt.ylabel('success rate', fontsize=16)
# plt.legend(fontsize=12) # Legend size
# plt.grid(alpha=0.3)

# plt.figure()
# p = plt.plot(sac_cr_eps, sac_cr, label='raw', alpha=ALPHA)
# plt.plot(sac_cr_eps, sac_cr_smoothed, label='smoothed, α=0.6', color=p[0].get_color())
# plt.xlabel('agents', fontsize=16)
# plt.ylabel('collision  rate', fontsize=16)
# plt.legend(fontsize=12) # Legend size
# plt.grid(alpha=0.3)

for i in range(len(sac_sr)):
    print('%2d\t%.2f\t%.2f' % (i+1,sac_sr[i]*100.0, sac_cr[i]*100.0))


plt.figure()
ax1 = plt.subplot()
plt.xlabel('total agents', fontsize=18)
l1_1 = ax1.plot(sac_sr_eps, sac_sr, label= r'$\textrm{raw}\ \mathit{SR}$', alpha=ALPHA)
l1_2 = ax1.plot(sac_sr_eps, sac_sr_smoothed, label=r'$\textrm{smoothed}\ \mathit{SR}$', color=l1_1[0].get_color())
plt.ylabel(r'$\textrm{Success Rate}\ (\mathit{SR})$', fontsize=18)
plt.legend(bbox_to_anchor=(0,0.8,1,0.2), loc="lower right", fontsize=18)
ax2 = ax1.twinx()
l2_1 = ax2.plot(sac_cr_eps, sac_cr, label= r'$\textrm{raw}\ \mathit{CR}$', alpha=ALPHA, color = 'orange')
l2_2 = ax2.plot(sac_cr_eps, sac_cr_smoothed, label=r'$\textrm{smoothed}\ \mathit{CR}$', color=l2_1[0].get_color())
plt.ylabel(r'$\textrm{Collision Rate}\ (\mathit{CR})$', fontsize=18)

plt.legend(fontsize=18)
plt.grid(alpha=0.3)
plt.legend(bbox_to_anchor=(0,0.2,1,0.2), loc="lower left", fontsize=18)

plt.show()


plt.show()