import torch
import numpy as np
from model import VDN
from buffer import ReplayBuffer
import torch.nn as nn
import torch.optim as optim
import random

class Agent():
    def __init__(self):
        self.buffer_size =5000
        self.batch_size = 32
        self.num_agents = 0
        self.num_of_actions = 8
        self.model = []
        self.buffer = []
        self.time = 0
        self.gamma = 0.9

    def init(self, obs):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)
        self.num_agents = len(obs['image'])
        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size, self.num_agents)
        self.model = VDN(self.num_agents, self.num_of_actions, self.device).to(self.device)
        self.target = VDN(self.num_agents, self.num_of_actions, self.device).to(self.device)
        self.update_target()
        self.optimizer = optim.Adam(self.model.parameters())
        self.last_state_cnn = np.zeros((self.num_agents,3,128,128))
        self.last_state_oth = np.zeros((self.num_agents, 11))
        self.last_action = np.zeros((self.num_agents, 1))

    def get_obs_cnn(self, obs):
        temp = []
        for i in range(len(obs["image"])):
            temp.append(np.r_[obs["image"][i]])
        temp = np.r_[temp]
        t = np.transpose(temp, (0,3,1,2)).astype(float)
        t /= 255.0
        return t

    def get_obs_oth(self, obs):
        temp = []
        # change in another network structure
        for i in range(len(obs["ir"])):
            temp.append(np.r_[obs["ir"][i],
                              obs["compass"][i],
                              obs["target"][i]])
        t = np.r_[temp]
        return t

    def get_new_cnn(self, t):
        t = np.concatenate((self.last_state_cnn, t), axis=1)
        return t

    def get_new_oth(self,t):
        t = np.concatenate((self.last_state_oth, t), axis=1)
        return t

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def get_action(self, obs, epsilon, done):
        if self.num_agents == 0:
            self.init(obs)
        state_cnn = self.get_obs_cnn(obs)
        state_oth = self.get_obs_oth(obs)
        cat_cnn = self.get_new_cnn(state_cnn)
        cat_oth = self.get_new_oth(state_oth)
        q, actions, _ = self.model(cat_cnn,cat_oth)
        index_action = np.zeros((self.num_agents,), dtype=np.uint8)
        for i in range(self.num_agents):
            if random.random() > epsilon:
                index_action[i] = random.randint(0, self.num_of_actions - 1)
            else:
                index_action[i] = actions[0,i].item()

        if done.item(0) != True:
            self.last_state_cnn = state_cnn
            self.last_state_oth = state_oth
            self.last_action = index_action
        elif done.item(0) == True:
            self.last_state_cnn = np.zeros((self.num_agents,3, 128, 128))
            self.last_state_oth = np.zeros((self.num_agents, 11))
            self.last_action = np.zeros((self.num_agents, 1))
        return index_action

    def learn(self):
        self.time += 1
        if len(self.buffer) < self.batch_size:
            return

        state_cnn, state_oth, action, reward, next_cnn, next_oth, done = self.buffer.sample()

        max_q = self.target(next_cnn, next_oth)[0].detach()
        _, _,  pred_q = self.model(state_cnn, state_oth, action)
        reward = reward.sum(1)
        true_q = reward + (1 - done[:,0]) * self.gamma * max_q
        criterion = nn.MSELoss()
        loss = criterion(pred_q, true_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.time % 10 == 0:
            self.update_target()

    def store_experience(self, obs, action, reward, done):
        state_cnn = self.get_obs_cnn(obs)
        state_oth = self.get_obs_oth(obs)
        self.buffer.add(state_cnn, state_oth, action, reward, done)
