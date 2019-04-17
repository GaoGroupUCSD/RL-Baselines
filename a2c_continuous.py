import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd import Variable
from itertools import count
import numpy as np
import math
import random
import os
import gym

# init a task generator for data fetching
env = gym.make('Pendulum-v0')

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
ACTION_MAX = env.action_space.high[0]
SAMPLE_NUMS = 1000

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor 
ByteTensor = torch.ByteTensor 
Tensor = FloatTensor

"""
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.1)
        m.bias.data.fill_(0.01)
"""

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)
        

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * std)
        
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp()
        dist = Normal(mu, std)
        return dist, value

class ActorNetwork(nn.Module):

    def __init__(self,state_dim, hidden_size,action_dim,std=0.0):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_dim)
        self.apply(init_weights)
        self.log_std = nn.Parameter(torch.ones(action_dim) * std)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        std = self.log_std.exp().expand_as(out)
        dist = Normal(out, std)
        return dist

class CriticNetwork(nn.Module):

    def __init__(self,state_dim,hidden_size,output_size=1):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,output_size)
        self.apply(init_weights)

    def forward(self,state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# init value network
critic_network = CriticNetwork(STATE_DIM,64,1)
critic_network_optim = torch.optim.Adam(critic_network.parameters(),lr=0.01)

# init actor network
actor_network = ActorNetwork(STATE_DIM,64,ACTION_DIM)
actor_network_optim = torch.optim.Adam(actor_network.parameters(),lr = 0.01)

model = ActorCritic(STATE_DIM, ACTION_DIM, 256)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

def roll_out(sample_nums):
    observation = env.reset()
    states = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    is_done = False
    final_r = 0
    episode_reward = 0
    entropy = 0
    for step in range(sample_nums):
        env.render()
        state = np.float32(observation)
        states.append(state)
        dist, value = model(Variable(torch.Tensor(state)))
        #dist = actor_network(Variable(torch.Tensor(state)))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()
        action = action.cpu().numpy()
        new_observation,reward,done,_ = env.step(action)
        episode_reward += reward
        log_probs.append(log_prob)
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        new_state = np.float32(new_observation)
        observation = new_observation
        if done:
            is_done = True
            break
    if not is_done:
        #final_r = critic_network(Variable(torch.Tensor(new_state)))
        _, final_r= model(Variable(torch.Tensor(new_state)))
    print ('REWARDS :- ', episode_reward)
    return states,actions,rewards,values,step,final_r,log_probs, entropy


def discount_reward(r, gamma,final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def update_network(states, actions, rewards, values, final_r, log_probs, entropy):
        log_probs = torch.cat(log_probs)
        vs = torch.cat(values)
        # calculate qs
        qs = Variable(torch.Tensor(discount_reward(rewards,0.99, final_r)))

        advantages = qs - vs
        actor_loss = - torch.mean(log_probs * advantages.detach())

        target_values = qs
        criterion = nn.MSELoss()
        critic_loss = criterion(vs,target_values)

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


MAX_EPISODES = 5000
MAX_STEPS = 1000


for _ep in range(MAX_EPISODES):
    observation = env.reset()
    print ('EPISODE :- ', _ep)
    states,actions,rewards,values,steps,final_r,log_probs, entropy = roll_out(SAMPLE_NUMS)
    update_network(states,actions,rewards,values,final_r,log_probs, entropy)