"""
Buffer system for the RL
"""
import threading
import numpy as np
import random
import torch
from torch import device, nn, tensor
from collections import deque
from torch.autograd import Variable

BUFFER_UNBALANCE_GAP = 0.5
USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

lock = threading.Lock()

class ReplayBuffer:
    """
    Replay Buffer to store the experiences.
    """

    def __init__(self, buffer_size, batch_size):
        """
        Initialize the attributes.

        Args:
            buffer_size: The size of the buffer memory
            batch_size: The batch for each of the data request `get_batch`
        """
        self.buffer = deque(maxlen=int(buffer_size))  # with format of (s,a,r,s')

        # constant sizes to use
        self.batch_size = batch_size

        # temp variables
        self.p_indices = [BUFFER_UNBALANCE_GAP/2]

    def append(self, state, action, r, sn, d):
        """
        Append to the Buffer

        Args:
            state: the state
            action: the action
            r: the reward
            sn: the next state
            d: done (whether one loop is done or not)
        """        
        
        state = to_tensor(state)
        action = to_tensor(action)  
        r = to_tensor(np.expand_dims(r, -1))
        sn = to_tensor(sn, True)
        d = to_tensor(np.expand_dims(d, -1))
        lock.acquire()
        self.buffer.append([state, action, r, sn, d])
        lock.release()

    def _append(self, state, action, r, sn, d):
        t = threading.Thread(target=self._append, args=(state, action, r, sn, d))
        t.start()

    def get_batch(self, unbalance_p=True):
        """
        Get the batch randomly from the buffer

        Args:
            unbalance_p: If true, unbalance probability of taking the batch from buffer with
            recent event being more prioritized

        Returns:
            the resulting batch
        """
        lock.acquire()
        # unbalance indices
        p_indices = None
        if random.random() < unbalance_p:
            self.p_indices.extend((np.arange(len(self.buffer)-len(self.p_indices))+1)
                                  * BUFFER_UNBALANCE_GAP + self.p_indices[-1])
            p_indices = self.p_indices / np.sum(self.p_indices)

        chosen_indices = np.random.choice(len(self.buffer),
                                          size=min(self.batch_size, len(self.buffer)),
                                          replace=False,
                                          p=p_indices)

        buffer = [self.buffer[chosen_index] for chosen_index in chosen_indices]
        lock.release()

        return buffer

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
        if volatile:
            with torch.no_grad():
                return Variable(
                    torch.from_numpy(ndarray), requires_grad=requires_grad
                ).type(dtype)        
        else:
            return Variable(
                    torch.from_numpy(ndarray), requires_grad=requires_grad
                ).type(dtype)        