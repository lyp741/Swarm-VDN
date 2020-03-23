import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


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
        self.adv1 = NoisyLinear(128,128)
        self.adv2 = NoisyLinear(128,num_actions)
        self.value1 = NoisyLinear(128,128)
        self.value2 = NoisyLinear(128,1)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)


    def forward(self, state_cnn, state_oth):
        state_cnn = torch.tensor(state_cnn).float().to(self.device)
        state_oth = torch.tensor(state_oth).float().to(self.device)
        state_cnn /= 255.0
        state_oth /= 255.0
        h = torch.tanh(self.bn1(self.conv1(state_cnn)))
        h2 = torch.tanh(self.bn2(self.conv2(h)))
        h3 = torch.tanh(self.bn3(self.conv3(h2)))
        h4 = torch.cat((h3.view(state_oth.shape[0],-1),state_oth),1)
        h4 = h4.view(state_cnn.shape[0],-1)
        # print('policy forward ',h4)
        h5 = F.relu(self.l1(h4))
        h6 = F.relu(self.l2(h5))
        value = F.relu(self.value1(h6))
        value = F.relu(self.value2(value))
        adv = F.relu(self.adv1(h6))
        adv = F.relu(self.adv2(adv))
        q = value + adv - adv.mean(dim=-1, keepdim=True)
        return q

    def reset_noise(self):
        # self.l1.reset_noise()
        # self.l2.reset_noise()
        self.adv1.reset_noise()
        self.adv2.reset_noise()
        self.value1.reset_noise()
        self.value2.reset_noise()

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
            action = torch.tensor(action).float().to(self.device)
            take_action = torch.zeros_like(action)
            for i in range(shaped.shape[0]):
                # print(shaped[i],action[i])
                a = action[i].unsqueeze(1).long()
                take_action[i] = torch.gather(shaped[i],1, a).squeeze()
            take_q = take_action.sum(1)
        q = max_q.sum(1)
        return q, actions, take_q

class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(self, in_features, out_features, std_init = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )
 
    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))

        return x.sign().mul(x.abs().sqrt())