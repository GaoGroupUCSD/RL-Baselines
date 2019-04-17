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

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, std=0.0):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * std)
        
        self.apply(init_weights)
        
    def forward(self, x):
        mu = self.actor(x)
        std = self.log_std.exp()
        dist = Normal(mu, std)
        return dist

# init actor network
actor_network = Actor(STATE_DIM,ACTION_DIM,256)
actor_network_optim = torch.optim.Adam(actor_network.parameters(),lr = 3e-4)
eps = np.finfo(np.float32).eps.item()

def roll_out(sample_nums):
    observation = env.reset()
    states = []
    actions = []
    rewards = []
    episode_reward = 0
    entropy = 0
    for _ in range(sample_nums):
        env.render()
        state = np.float32(observation)
        states.append(state)
        dist = actor_network(Variable(torch.Tensor(state)))
        print("mean", dist.loc)
        print("std",dist.scale)
        action = dist.sample()
        entropy += dist.entropy().mean()
        action = action.cpu().numpy()
        new_observation,reward,done,_ = env.step(action)
        episode_reward += reward
        actions.append(action)
        rewards.append(reward)
        observation = new_observation
        if done:
            break
    print ('REWARDS :- ', episode_reward)
    return states,actions,rewards,entropy

def discount_reward(r, gamma):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def update_network(states, actions, rewards, entropy):
    states_var = Variable(FloatTensor(states).view(-1,STATE_DIM))
    actions_var = Variable(FloatTensor(actions).view(-1,ACTION_DIM))
    # train actor network
    actor_network_optim.zero_grad()
    dist = actor_network(states_var)
    log_probs = dist.log_prob(actions_var)
    # calculate qs
    rewards = Variable(torch.Tensor(discount_reward(rewards,0.99)))
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    actor_network_loss = - torch.mean(torch.sum(log_probs * rewards))
    #print("loss",actor_network_loss)
    actor_network_loss.backward()
    actor_network_optim.step()

MAX_EPISODES = 5000
MAX_STEPS = 1000


for _ep in range(MAX_EPISODES):
    observation = env.reset()
    print ('EPISODE :- ', _ep)
    states,actions,rewards,entropy = roll_out(SAMPLE_NUMS)
    update_network(states,actions,rewards,entropy)