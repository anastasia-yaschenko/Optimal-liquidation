import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from DataHandler import DataHandler
from TradingEnvDiscrete import TradingEnvironment
from TradingSimulatorDiscrete import TradingSimulator
# dynamics params
X_0 = 100000
q = 5000
F_0 = 1000 # init price
rho = 2.2231
lambda_ = 1/(2*q)
T = 1
N = 150
timestamps = list(range(0, N+1))#[n*T/N for n in range(0,N+1)]

def get_random_walk(scale_param = 5):
    P = F_0
    F = [P]
    for i in range(N+1):
        P += np.random.normal(0, 1)*scale_param
        F.append(P)
    return F
F = get_random_walk()
plt.plot(F)
data = pd.Series(F)

env = TradingEnvironment(TradingSimulator, data, len(data))

#https://github.com/BY571/SAC_discrete

import gym
import numpy as np
from collections import deque
import torch
from buffer import ReplayBuffer
import glob
from utils import save, collect_random
import random
from agent import SAC


np.random.seed(1)
random.seed(1)
torch.manual_seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

steps = 0
average10 = deque(maxlen=10)
total_steps = 0
    

agent = SAC(state_size=3, action_size=101, device=device)

buffer = ReplayBuffer(buffer_size=10000, batch_size=64, device=device)

collect_random(env=env, dataset=buffer, num_samples=10000)


episodes = 10
for episode in range(episodes):
  state = env.reset()
  episode_steps = 0
  rewards = 0
  for s in range(len(data)):
      action = agent.get_action(state)
      steps += 1
      next_state, reward, info, done  = env.step(np.floor(action/100*env.simulator.remaining))
      buffer.add(state, action, reward, next_state, done)
      policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = agent.learn(steps, buffer.sample(), gamma=1)
      state = next_state
      rewards += reward
      episode_steps += 1
      if done:
            pass

  if episode % 1 == 0: # print average shortfall over last 100 episodes
        print('Episode:', episode, 'Average episode reward', np.mean(env.simulator.rewards), 'Total episode reward', np.sum(env.simulator.rewards))