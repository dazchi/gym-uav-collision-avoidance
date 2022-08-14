

import csv
from turtle import color
import matplotlib.pyplot as plt
import pandas as pd

SMOOTH_FACTOR = 0.8
ALPHA = 0.3

df_sac_sr = pd.read_csv('./SAC/SR.csv') # Path of csv file
df_sac_cr = pd.read_csv('./SAC/CR.csv') # Path of csv file
df_sac_sr_smoothed = df_sac_sr.ewm(alpha=(1 - SMOOTH_FACTOR)).mean()
df_sac_cr_smoothed = df_sac_cr.ewm(alpha=(1 - SMOOTH_FACTOR)).mean()
sac_sr_eps = df_sac_sr['Step'].values.tolist()
sac_sr = df_sac_sr['Value'].values.tolist()
sac_sr_smoothed = df_sac_sr_smoothed['Value'].values.tolist()
sac_cr_eps = df_sac_cr['Step'].values.tolist()
sac_cr = df_sac_cr['Value'].values.tolist()
sac_cr_smoothed = df_sac_cr_smoothed['Value'].values.tolist()

df_td3_sr = pd.read_csv('./TD3/SR.csv') # Path of csv file
df_td3_cr = pd.read_csv('./TD3/CR.csv') # Path of csv file
df_td3_sr_smoothed = df_td3_sr.ewm(alpha=(1 - SMOOTH_FACTOR)).mean()
df_td3_cr_smoothed = df_td3_cr.ewm(alpha=(1 - SMOOTH_FACTOR)).mean()
td3_sr_eps = df_td3_sr['Step'].values.tolist()
td3_sr = df_td3_sr['Value'].values.tolist()
td3_sr_smoothed = df_td3_sr_smoothed['Value'].values.tolist()
td3_cr_eps = df_td3_cr['Step'].values.tolist()
td3_cr = df_td3_cr['Value'].values.tolist()
td3_cr_smoothed = df_td3_cr_smoothed['Value'].values.tolist()


df_ddpg_sr = pd.read_csv('./DDPG/SR.csv') # Path of csv file
df_ddpg_cr = pd.read_csv('./DDPG/CR.csv') # Path of csv file
df_ddpg_sr_smoothed = df_ddpg_sr.ewm(alpha=(1 - SMOOTH_FACTOR)).mean()
df_ddpg_cr_smoothed = df_ddpg_cr.ewm(alpha=(1 - SMOOTH_FACTOR)).mean()
ddpg_sr_eps = df_ddpg_sr['Step'].values.tolist()
ddpg_sr = df_ddpg_sr['Value'].values.tolist()
ddpg_sr_smoothed = df_ddpg_sr_smoothed['Value'].values.tolist()
ddpg_cr_eps = df_ddpg_cr['Step'].values.tolist()
ddpg_cr = df_ddpg_cr['Value'].values.tolist()
ddpg_cr_smoothed = df_ddpg_cr_smoothed['Value'].values.tolist()



plt.figure()
p = plt.plot(sac_sr_eps, sac_sr, alpha=ALPHA)
plt.plot(sac_sr_eps, sac_sr_smoothed, label='SAC', color=p[0].get_color())

p = plt.plot(td3_sr_eps, td3_sr, alpha=ALPHA)
plt.plot(td3_sr_eps, td3_sr_smoothed, label='TD3', color=p[0].get_color())

p = plt.plot(ddpg_sr_eps, ddpg_sr, alpha=ALPHA)
plt.plot(ddpg_sr_eps, ddpg_sr_smoothed, label='DDPG', color=p[0].get_color())

plt.xlabel('epsiodes')
plt.ylabel('success rate')
plt.legend(fontsize=12) # Legend size
plt.grid(alpha=0.3)

plt.figure()
p = plt.plot(sac_cr_eps, sac_cr, alpha=ALPHA)
plt.plot(sac_cr_eps, sac_cr_smoothed, label='SAC', color=p[0].get_color())

p = plt.plot(td3_cr_eps, td3_cr, alpha=ALPHA)
plt.plot(td3_cr_eps, td3_cr_smoothed, label='TD3', color=p[0].get_color())

p = plt.plot(ddpg_cr_eps, ddpg_cr, alpha=ALPHA)
plt.plot(ddpg_cr_eps, ddpg_cr_smoothed, label='DDPG', color=p[0].get_color())

plt.xlabel('epsiodes')
plt.ylabel('collision rate')
plt.legend(fontsize=12) # Legend size
plt.grid(alpha=0.3)



plt.show()