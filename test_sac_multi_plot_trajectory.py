
import colorsys
from copy import deepcopy
import torch
import time
import math
import numpy as np
from gym_uav_collision_avoidance.envs import MultiUAVWorld2D
from pytorch_sac_temp.sac import SAC
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

MODEL_PATH = './weights/sac_multi'
N_AGENT = 12
MAX_EPISOED_STEPS = 2000
OFFSET = 2
RANDOM = False


# If GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device = %s' % device)

trajectories = []
depots = []
goals = []



env = MultiUAVWorld2D(num_agents=N_AGENT)
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
agent = SAC(n_observations, n_actions)
agent.load_checkpoint(MODEL_PATH, evaluate=True)

for i in range(N_AGENT):
    trajectories.append([])

time.sleep(1)
states, _ = env.reset(return_info=True)


for i in range(N_AGENT):
    if not RANDOM:
        theta = 2 * i * math.pi / N_AGENT
        env.agent_list[i].location = 20 * np.ones(2) * np.array([math.cos(theta), math.sin(theta)])
        env.agent_list[i].target_location = 23 * np.ones(2) * np.array([math.cos(theta+math.pi-OFFSET*math.pi/ N_AGENT), math.sin(theta+math.pi-OFFSET*math.pi/ N_AGENT)])
    depots.append(deepcopy(env.agent_list[i].location))
    goals.append(env.agent_list[i].target_location)

for steps in range(MAX_EPISOED_STEPS):
    actions = []
    converted_actions = []

    for i in range(N_AGENT):
        if env.agent_list[i].done:
            converted_actions.append(np.zeros(2))
            continue
        action = agent.select_action(states[i], evaluate=True)                            
        v = (action[0]/2+0.5) * np.linalg.norm(env.action_space.high)     
        theta = action[1] * math.pi            
        converted_action = np.array([v*math.cos(theta), v*math.sin(theta)])                
        actions.append(action)
        converted_actions.append(converted_action)
        trajectories[i].append(deepcopy(env.agent_list[i].location))
    
    next_states, _, dones, _ = env.step(converted_actions) # Step                   
    states = next_states
    env.render()       
    
    if all(dones):
        break         

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"

colors = []
# colors = cm.rainbow(np.linspace(0, 1, N_AGENT))
for i in range(N_AGENT):
    hue = i / N_AGENT
    (r, g, b) = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  
    colors.append((r, g, b))

for i in range(N_AGENT):
    xs, ys = np.array(trajectories[i])[:,0],np.array(trajectories[i])[:,1]
    l = plt.plot(xs,ys,'-', color=colors[i])
    # colors.append(l[0].get_color())

xs, ys = np.array(depots)[:,0],np.array(depots)[:,1]
for i in range(N_AGENT):        
    x, y = xs[i], ys[i]
    plt.plot(x, y,'^', color=colors[i], markersize=10)        
    plt.text(x+0.5, y+0.3, r'$\mathit{D}_{%d}$'%i, color='black', fontsize=10)

xs, ys = np.array(goals)[:,0],np.array(goals)[:,1]
for i in range(N_AGENT):            
    x, y = xs[i], ys[i]                         
    plt.plot(x, y,'s', color=colors[i])
    plt.text(x+0.5, y+0.3, r'$\mathit{G}_{%d}$'%i, color='black', fontsize=10)
    

plt.xlabel(r'$\mathit{x}\textrm{-coordinate}\ (\mathrm{m})$', fontsize=14)
plt.ylabel(r'$\mathit{y}\textrm{-coordinate}\ (\mathrm{m})$', fontsize=14)



triangle = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                          markersize=10, label='Depots', fillstyle='none')
square = mlines.Line2D([], [], color='black', marker='s', linestyle='None',
                          markersize=10, label='Goals', fillstyle='none')

plt.legend(handles=[triangle, square])
plt.show()

        