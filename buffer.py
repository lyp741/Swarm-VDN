from collections import deque, namedtuple
import random
import torch
import numpy as np
class ReplayBuffer:
    " Internal memory of the agent "
    
    def __init__(self, buffer_size, batch_size, n_agents=1, seed=0):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)
        self.n_agents = n_agents
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
        self.experience = namedtuple("Experience", field_names=["state_cnn", "state_oth", "action", "reward", "done"])
        
        self.seed = random.seed(seed)
        
    def add(self, state_cnn, state_oth, action, reward, done):
        " Add a new experience to memory "
        # for i in range(self.n_agents):
        e = self.experience(state_cnn, state_oth, action, reward, done)
        self.memory.append(e)
        
    def sample(self):
        " Randomly sample a batch of experiences from the memory "
        rand_idx = np.random.randint(1,len(self.memory)-1,self.batch_size)
        next_rand_idx = rand_idx + 1
        prev_rand_idx = rand_idx - 1

        states = [self.memory[i] for i in rand_idx]
        next_states = [self.memory[i] for i in next_rand_idx]
        prev_states = [self.memory[i] for i in prev_rand_idx]

        prev_states_cnn = torch.from_numpy(np.vstack([e.state_cnn for e in prev_states if e is not None])).float()
        prev_states_oth = torch.from_numpy(np.vstack([e.state_oth for e in prev_states if e is not None])).float()
        states_cnn = torch.from_numpy(np.vstack([e.state_cnn for e in states if e is not None])).float()
        states_oth = torch.from_numpy(np.vstack([e.state_oth for e in states if e is not None])).float()
        next_states_cnn = torch.from_numpy(np.vstack([e.state_cnn for e in next_states if e is not None])).float()
        next_states_oth = torch.from_numpy(np.vstack([e.state_oth for e in next_states if e is not None])).float()

        actions = torch.from_numpy(np.vstack([e.action for e in states if e is not None])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in states if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in states if e is not None]).astype(np.uint8)).float()

        next_states_cnn = torch.cat((states_cnn, next_states_cnn), 1)
        next_states_oth = torch.cat((states_oth,next_states_oth), 1)
        states_cnn = torch.cat((prev_states_cnn, states_cnn), 1)
        states_oth = torch.cat((prev_states_oth, states_oth), 1)
        return states_cnn.to(self.device), states_oth.to(self.device), actions.to(self.device), rewards.to(self.device), next_states_cnn.to(self.device), next_states_oth.to(self.device), dones.to(self.device)
    
    def __len__(self):
        " Return the current size of internal memory. Overwrites the inherited function len. "
        
        return len(self.memory)
        
    