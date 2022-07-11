import os
import torch
import numpy as np
from ddpg import DDPG



# If GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device = %s' % device)

ddpg = DDPG(5,2)






