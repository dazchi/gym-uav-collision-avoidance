
import matplotlib.pyplot as plt
import matplotlib.font_manager
import pandas as pd

SMOOTH_FACTOR = 0.9
ALPHA = 0.3
EPS = 150

df_sac_sr = pd.read_csv('./logs/SAC/SR.csv') # Path of csv file
df_sac_cr = pd.read_csv('./logs/SAC/CR.csv') # Path of csv file
df_sac_score = pd.read_csv('./logs/SAC/Score.csv') # Path of csv file
df_sac_sr_smoothed = df_sac_sr.ewm(alpha=(1 - SMOOTH_FACTOR)).mean()
df_sac_cr_smoothed = df_sac_cr.ewm(alpha=(1 - SMOOTH_FACTOR)).mean()
df_sac_score_smoothed = df_sac_score.ewm(alpha=(1 - SMOOTH_FACTOR)).mean()
sac_sr_eps = df_sac_sr['Step'].values.tolist()
sac_sr = df_sac_sr['Value'].values.tolist()
sac_sr_smoothed = df_sac_sr_smoothed['Value'].values.tolist()
sac_cr_eps = df_sac_cr['Step'].values.tolist()
sac_cr = (df_sac_cr['Value'].values).tolist()
sac_cr_smoothed = (df_sac_cr_smoothed['Value'].values).tolist()
sac_score_eps = df_sac_score['Step'].values.tolist()
sac_score = df_sac_score['Value'].values.tolist()
sac_score_smoothed = df_sac_score_smoothed['Value'].values.tolist()

df_td3_sr = pd.read_csv('./logs/TD3/SR_1500.csv') # Path of csv file
df_td3_cr = pd.read_csv('./logs/TD3/CR_1500.csv') # Path of csv file
df_td3_score = pd.read_csv('./logs/TD3/Score_1500.csv') # Path of csv file
df_td3_sr_smoothed = df_td3_sr.ewm(alpha=(1 - SMOOTH_FACTOR)).mean()
df_td3_cr_smoothed = df_td3_cr.ewm(alpha=(1 - SMOOTH_FACTOR)).mean()
df_td3_score_smoothed = df_td3_score.ewm(alpha=(1 - SMOOTH_FACTOR)).mean()
td3_sr_eps = df_td3_sr['Step'].values.tolist()
td3_sr = df_td3_sr['Value'].values.tolist()
td3_sr_smoothed = df_td3_sr_smoothed['Value'].values.tolist()
td3_cr_eps = df_td3_cr['Step'].values.tolist()
td3_cr = df_td3_cr['Value'].values.tolist()
td3_cr_smoothed = df_td3_cr_smoothed['Value'].values.tolist()
td3_score_eps = df_td3_score['Step'].values.tolist()
td3_score = df_td3_score['Value'].values.tolist()
td3_score_smoothed = df_td3_score_smoothed['Value'].values.tolist()


df_ddpg_sr = pd.read_csv('./logs/DDPG/SR.csv') # Path of csv file
df_ddpg_cr = pd.read_csv('./logs/DDPG/CR.csv') # Path of csv file
df_ddpg_score = pd.read_csv('./logs/DDPG/Score.csv') # Path of csv file
df_ddpg_sr_smoothed = df_ddpg_sr.ewm(alpha=(1 - SMOOTH_FACTOR)).mean()
df_ddpg_cr_smoothed = df_ddpg_cr.ewm(alpha=(1 - SMOOTH_FACTOR)).mean()
df_ddpg_score_smoothed = df_ddpg_score.ewm(alpha=(1 - SMOOTH_FACTOR)).mean()
ddpg_sr_eps = df_ddpg_sr['Step'].values.tolist()
ddpg_sr = df_ddpg_sr['Value'].values.tolist()
ddpg_sr_smoothed = df_ddpg_sr_smoothed['Value'].values.tolist()
ddpg_cr_eps = df_ddpg_cr['Step'].values.tolist()
ddpg_cr = df_ddpg_cr['Value'].values.tolist()
ddpg_cr_smoothed = df_ddpg_cr_smoothed['Value'].values.tolist()
ddpg_score_eps = df_ddpg_score['Step'].values.tolist()
ddpg_score = df_ddpg_score['Value'].values.tolist()
ddpg_score_smoothed = df_ddpg_score_smoothed['Value'].values.tolist()

sac_sr_eps = sac_sr_eps[:EPS]
sac_sr = sac_sr [:EPS]
sac_sr_smoothed = sac_sr_smoothed[:EPS]
sac_cr_eps = sac_cr_eps[:EPS]
sac_cr = sac_cr[:EPS]
sac_cr_smoothed = sac_cr_smoothed[:EPS]

t_eps = 600 -12
sac_score_eps = sac_score_eps[:t_eps]
sac_score = sac_score[:t_eps]
sac_score_smoothed = sac_score_smoothed[:t_eps]

td3_sr_eps = td3_sr_eps[:EPS]
td3_sr = td3_sr [:EPS]
td3_sr_smoothed = td3_sr_smoothed[:EPS]
td3_cr_eps = td3_cr_eps[:EPS]
td3_cr = td3_cr[:EPS]
td3_cr_smoothed = td3_cr_smoothed[:EPS]
td3_score_eps = td3_score_eps[:EPS*10]
td3_score = td3_score[:EPS*10]
td3_score_smoothed = td3_score_smoothed[:EPS*10]

ddpg_sr_eps = ddpg_sr_eps[:EPS]
ddpg_sr = ddpg_sr [:EPS]
ddpg_sr_smoothed = ddpg_sr_smoothed[:EPS]
ddpg_cr_eps = ddpg_cr_eps[:EPS]
ddpg_cr = ddpg_cr[:EPS]
ddpg_cr_smoothed = ddpg_cr_smoothed[:EPS]
ddpg_score_eps = ddpg_score_eps[:EPS*10]
ddpg_score = ddpg_score[:EPS*10]
ddpg_score_smoothed = ddpg_score_smoothed[:EPS*10]


# fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
# for ft in fonts:
#     if 'Time' in ft:
#         print(ft)

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
plt.figure()
p = plt.plot(sac_sr_eps, sac_sr, alpha=ALPHA)
plt.plot(sac_sr_eps, sac_sr_smoothed, label='SAC', color=p[0].get_color())

p = plt.plot(td3_sr_eps, td3_sr, alpha=ALPHA)
plt.plot(td3_sr_eps, td3_sr_smoothed, label='TD3', color=p[0].get_color())

p = plt.plot(ddpg_sr_eps, ddpg_sr, alpha=ALPHA)
plt.plot(ddpg_sr_eps, ddpg_sr_smoothed, label='DDPG', color=p[0].get_color())

plt.xlabel('epsiodes', fontsize=16)
plt.ylabel(r'$\textrm{Success Rate}\ (\mathit{SR})$', fontsize=16)
plt.legend(fontsize=12) # Legend size
plt.grid(alpha=0.3)

plt.figure()
p = plt.plot(sac_cr_eps, sac_cr, alpha=ALPHA)
plt.plot(sac_cr_eps, sac_cr_smoothed, label='SAC', color=p[0].get_color())

p = plt.plot(td3_cr_eps, td3_cr, alpha=ALPHA)
plt.plot(td3_cr_eps, td3_cr_smoothed, label='TD3', color=p[0].get_color())

p = plt.plot(ddpg_cr_eps, ddpg_cr, alpha=ALPHA)
plt.plot(ddpg_cr_eps, ddpg_cr_smoothed, label='DDPG', color=p[0].get_color())

plt.xlabel('epsiodes', fontsize=16)
plt.ylabel(r'$\textrm{Collision Rate}\ (\mathit{CR})$', fontsize=16)
plt.legend(fontsize=12) # Legend size
plt.grid(alpha=0.3)

plt.figure()
p = plt.plot(sac_score_eps, sac_score, label='raw', alpha=ALPHA)
plt.plot(sac_score_eps, sac_score_smoothed, label=r'$\textrm{smoothed}$', color=p[0].get_color())

# p = plt.plot(td3_score_eps, td3_score, alpha=ALPHA)
# plt.plot(td3_score_eps, td3_score_smoothed, label='TD3', color=p[0].get_color())

# p = plt.plot(ddpg_score_eps, ddpg_score, alpha=ALPHA)
# plt.plot(ddpg_score_eps, ddpg_score_smoothed, label='DDPG', color=p[0].get_color())

plt.xlabel('epsiodes', fontsize=16)
plt.ylabel('score', fontsize=16)
plt.legend(fontsize=12) # Legend size
plt.grid(alpha=0.3)


plt.show()