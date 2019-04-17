import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
from itertools import count
import numpy as np
import math
import random
import os
import gym

## Hyper Parameters
STATE_DIM = 4
ACTION_DIM = 2
SAMPLE_NUMS = 1000

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor 
ByteTensor = torch.ByteTensor 
Tensor = FloatTensor

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class ActorNetwork(nn.Module):

    def __init__(self,state_dim,action_dim,hidden_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_dim)
        self.apply(init_weights)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out), dim=1)
        dist = Categorical(out)
        return dist

class ValueNetwork(nn.Module):

    def __init__(self,state_dim,action_dim,hidden_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_dim)
        self.apply(init_weights)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# init value network
value_network = ValueNetwork(STATE_DIM,1,64)
value_network_optim = torch.optim.Adam(value_network.parameters(),lr=0.001)

# init actor network
actor_network = ActorNetwork(STATE_DIM,ACTION_DIM,64)
actor_network_optim = torch.optim.Adam(actor_network.parameters(),lr = 0.001)

# init a task generator for data fetching
env = gym.make("CartPole-v0")

def roll_out(sample_nums):
    state = env.reset()
    states = []
    actions = []
    rewards = []
    is_done = False
    final_r = 0
    for step in range(sample_nums):
        states.append(state)
        dist = actor_network(Variable(torch.Tensor([state])))
        action = dist.sample()
        actions.append(action)
        action = action.cpu().numpy()
        next_state,reward,done,_ = env.step(action[0])
        rewards.append(reward)
        state = next_state
        if done:
            is_done = True
            break
    if not is_done:
        final_r = value_network(Variable(torch.Tensor([state])))
    return states,actions,rewards,step,final_r


def discount_reward(r, gamma, final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


