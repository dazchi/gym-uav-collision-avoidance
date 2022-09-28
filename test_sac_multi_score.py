import time
import torch
import math
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from gym_uav_collision_avoidance.envs import MultiUAVWorld2D
from pytorch_sac_temp.sac import SAC


MODEL_PATH = './weights/sac_multi'
TEST_EPISODES = 100
MAX_NUM_AGENTS = 24
MAX_EPISOED_STEPS = 2000


# If GPU is to be used

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device = %s' % device)
comment = "evaluate %"
tb_writer = SummaryWriter(comment=comment)

agents = []
dummy_env = MultiUAVWorld2D(num_agents=1)
n_observations = dummy_env.observation_space.shape[0]
n_actions = dummy_env.action_space.shape[0]
for i in range(MAX_NUM_AGENTS):
    agents.append(SAC(n_observations, n_actions))
    agents[i].load_checkpoint(MODEL_PATH, evaluate=True)

for n_agnet in range(1,MAX_NUM_AGENTS+1):
    env = MultiUAVWorld2D(num_agents=n_agnet)
          
    success_count = 0
    collision_count = 0
    states, _ = env.reset(return_info=True)
    for eps in range(TEST_EPISODES):    
        score = 0
        eps_t = time.time()
        eps_steps = 0
        for steps in range(MAX_EPISOED_STEPS):
            actions = []
            converted_actions = []

            for i in range(n_agnet):                
                action = agents[i].select_action(states[i], evaluate=True)                            
                v = (action[0]/2+0.5) * np.linalg.norm(env.action_space.high)     
                theta = action[1] * math.pi            
                converted_action = np.array([v*math.cos(theta), v*math.sin(theta)])                
                actions.append(action)
                converted_actions.append(converted_action)
            
            next_states, rewards, dones, _ = env.step(converted_actions, evaluate=True) # Step                   
            states = next_states
            score += rewards[0]
            eps_steps += 1            
            env.render()       
            
            if all(dones):
                break         
        
        
        success_count += env.target_reach_count
        collision_count += env.collision_count     
        sr = success_count / (n_agnet * (eps+1))
        cr = collision_count / (n_agnet * (eps+1))        
        states, _ = env.reset(return_info=True)
        eps_t = time.time() - eps_t
        steps_per_sec = eps_steps / eps_t        
        print("Score = %.3f, Steps Per Sec = %.2f" % (score, steps_per_sec))    
        print("N = %d, EP = %d, SR = %.3f, CR = %.3f, Steps = %d" % (n_agnet, eps, sr, cr, eps_steps))
    
    print(success_count)
    print(collision_count)
    success_rate = success_count / (n_agnet * TEST_EPISODES)
    collision_rate = collision_count / (n_agnet * TEST_EPISODES)
    print("SR = %.2f, CR = %.2f" % (success_rate, collision_rate)) 
    tb_writer.add_scalar("SR/AGENTS", success_rate, n_agnet)
    tb_writer.add_scalar("CR/AGENTS", collision_rate, n_agnet)
    tb_writer.flush()

env.close()
tb_writer.close()





        