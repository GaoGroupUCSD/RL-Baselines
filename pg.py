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

# init a task generator for data fetching
env = gym.make("CartPole-v0")

## Hyper Parameters
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n
SAMPLE_NUMS = 1000

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor 
ByteTensor = torch.ByteTensor 
Tensor = FloatTensor

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

# init actor network
actor_network = ActorNetwork(STATE_DIM,ACTION_DIM,64)
actor_network_optim = torch.optim.Adam(actor_network.parameters(),lr = 0.001)
eps = np.finfo(np.float32).eps.item()

def roll_out(sample_nums):
    state = env.reset()
    states = []
    actions = []
    rewards = []

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
            break

    return states,actions,rewards,step

def update_network(states, actions, rewards):
        actions_var = torch.cat(actions)
        states_var = Variable(FloatTensor(states).view(-1,STATE_DIM))
        # train actor network
        actor_network_optim.zero_grad()
        dist = actor_network(states_var)
        log_probs = dist.log_prob(actions_var)
        # calculate qs
        rewards = Variable(torch.Tensor(discount_reward(rewards,0.99)))
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        actor_network_loss = - torch.mean(torch.sum(log_probs * rewards))
        actor_network_loss.backward()
        actor_network_optim.step()


def discount_reward(r, gamma):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def main():
    running_reward = 10
    print("reward threshold", env.spec.reward_threshold)

    for i_episode in count(1):
        states,actions,rewards,steps = roll_out(SAMPLE_NUMS)
        running_reward = running_reward * 0.99 + steps * 0.01
        update_network(states,actions,rewards)
        
        if i_episode % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, steps+1, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, steps+1))
            break
    # test
    for i_episode in range(10):
        state = env.reset()
        for t in range(1000):
            env.render()
            dist = actor_network(FloatTensor([state]))
            action = dist.sample()
            state, reward, done, info = env.step(action.cpu().numpy()[0])
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    env.close()

if __name__ == '__main__':
    main()