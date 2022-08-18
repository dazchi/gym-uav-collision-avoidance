import time
import sys
import numpy as np
from pytorch_sac_temp.sac import SAC

MODEL_PATH = './weights/sac_multi'
TEST_TIME = 10000

agent = SAC(10, 2)
agent.load_checkpoint(MODEL_PATH)

observation = np.random.rand(10) * 2 - 1
agent.select_action(observation)

total_t = 0
for i in range(TEST_TIME):
    observation = np.random.rand(10) * 2 - 1        
    t = time.time()
    agent.select_action(observation)
    delta_t = time.time() - t
    total_t += delta_t
    print(delta_t, end='\r')
sys.stdout.write("\033[K")
print("agv = %.3f", total_t/TEST_TIME)