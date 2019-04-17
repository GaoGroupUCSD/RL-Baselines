## RLalgorithms

RL agents using various reinforcement learning algorithms.
Test mainly on OpenAI gym environments.

1. Advantage Actor Critic (A2C)
- Discrete action space version
- [a2c.py](https://github.com/GaoGroupUCSD/RLalgorithms/blob/master/a2c.py)
- [a3c.py](https://github.com/GaoGroupUCSD/RLalgorithms/blob/master/a3c.py)
- [A3C Paper](https://arxiv.org/abs/1602.01783) 
2.  Proximal Policy Optimization 
- Discrete action space version
- [ppo.py](https://github.com/GaoGroupUCSD/RLalgorithms/blob/master/ppo.py)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
3. Deep Deterministic Policy Gradient
- Using replay memory and ornstein uhlenbeck noise
- [ddpg.py](https://github.com/GaoGroupUCSD/RLalgorithms/blob/master/ddpg.py)
- [DDPG Paper](https://arxiv.org/abs/1509.02971)
4. Deep Q Learning
- Using replay memory and asynchronous update
- [dqn.py](https://github.com/GaoGroupUCSD/RLalgorithms/blob/master/dqn.py)
- [DQN Paper](https://arxiv.org/abs/1312.5602)
5. Policy Gradient
- Discrete action space version
- [pg.py](https://github.com/GaoGroupUCSD/RLalgorithms/blob/master/pg.py)
- [PG Blog](http://karpathy.github.io/2016/05/31/rl/)