import time
import sys
import math
import numpy as np
from pytorch_sac_temp.sac import SAC

MODEL_PATH = './weights/sac_multi'
TEST_TIME = 10000

agent = SAC(10, 2, cpu=True)
agent.load_checkpoint(MODEL_PATH)

observation = np.random.rand(10) * 2 - 1
agent.select_action(observation)

total_t = 0
data = []
for i in range(TEST_TIME):
    observation = np.random.rand(10) * 2 - 1        
    t = time.time()
    agent.select_action(observation)
    delta_t = time.time() - t
    total_t += delta_t
    data.append(delta_t)
    print(delta_t, end='\r')
sys.stdout.write("\033[K")

square_mean = 0
for i in range(TEST_TIME):
    square_mean += math.pow(data[i],2)

square_mean /= TEST_TIME
mean = total_t / TEST_TIME
sigma = math.sqrt(square_mean - math.pow(mean,2))
mean *= 1000
sigma *= 1000
max = 1000 * max(data)
min = 1000 * min(data)

print("agv = %.3f, sigma = %.3f, max = %.3f, min = %.3f" % (mean, sigma, max, min))