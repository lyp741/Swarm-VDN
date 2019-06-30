import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, num_actions):
        super(Policy,self).__init__()
        self.conv1 = nn.Conv2d(6,32,4,2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,32,4,2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,32,4,2)
        self.bn3 = nn.BatchNorm2d(32)
        self.l1 = nn.Linear(6294,128)
        self.l2 = nn.Linear(128,128)
        self.q = nn.Linear(128, num_actions)

    def forward(self, state_cnn, state_oth):

        h = torch.tanh(self.bn1(self.conv1(state_cnn)))
        h2 = torch.tanh(self.bn2(self.conv2(h)))
        h3 = torch.tanh(self.bn3(self.conv3(h2)))
        h4 = torch.cat((h3.view(state_oth.shape[0],-1),state_oth),1)
        # print('policy forward ',h4)
        h5 = F.relu(self.l1(h4))
        h6 = F.relu(self.l2(h5))
        q = self.q(h6)
        return q

class VDN(nn.Module):
    def __init__(self, num_agents, num_actions, device):
        super(VDN, self).__init__()
        self.num_actions = num_actions
        self.policy = Policy(num_actions)
        self.num_agents = num_agents
        self.device = device

    def forward(self, state_cnn, state_oth, action=None):
        take_q = None
        state_cnn = torch.tensor(state_cnn).float().to(self.device)
        state_oth = torch.tensor(state_oth).float().to(self.device)
        o = self.policy(state_cnn,state_oth)
        shaped = o.view(int(o.shape[0]/self.num_agents), self.num_agents, self.num_actions)
        max_q, actions = torch.max(shaped, 2)
        if action is not None:
            take_action = torch.zeros_like(action)
            for i in range(shaped.shape[0]):
                # print(shaped[i],action[i])
                a = action[i].unsqueeze(1).long()
                take_action[i] = torch.gather(shaped[i],1, a).squeeze()
            take_q = take_action.sum(1)
        q = max_q.sum(1)
        return q, actions, take_q
