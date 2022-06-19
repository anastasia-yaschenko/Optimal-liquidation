import numpy as np
from math import isclose
import gym

from rsa import verify
class TradingSimulator:

    def __init__(self, datahandler, steps):
        self.datahandler = datahandler
        self.steps = steps #N+2
        #self.N = steps - 2
        self.step = 0
        self.prices = np.zeros(self.steps)
        self.actions = np.zeros(self.steps)
        self.execution_prices = np.zeros(self.steps)
        self.strategy_costs = np.zeros(self.steps)
        self.inventory = 0
        self.X_0 = 1
        self.rewards = np.zeros(self.steps)



        # GYM wrappers
        self.action_space = gym.spaces.Box(0.0, 1.0, (1,))
        self.observation_space =  gym.spaces.Box(0.0, 1.0, (1,))

    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.prices.fill(0)
        self.rewards.fill(0)
        self.prices[0] = self.datahandler.data[0]
        self.strategy_costs.fill(0)
        self.inventory = 0
        self.remaining = self.X_0 - self.inventory
        self.execution_prices.fill(0)
        return np.array([
                       # 0,
                        np.log(self.prices[0]),
                        self.remaining/self.X_0,
                        (self.N - self.step)/self.N
                        ])
    
    def verify_action(self, action):
        return True

    def take_verified_step(self, action):
        if self.step == self.steps-1 and isclose(self.remaining, 0): # self.steps - 1 technical ending
            
            action = 0
            reward = 0
            
            self.execution_prices[self.step] = self.prices[self.step]
            self.strategy_costs[self.step] = reward

            self.prices[self.step] = self.datahandler.data[self.step]
            
            state = np.array([
                #    -1,
                    -1,
                    -1,
                    -1
                ])
            info = self.prices
            self.step += 1
            if self.step >= self.steps:
                done = True
            else:
                done = False
            return reward, state, info, done

        if self.step == self.steps-1 and not isclose(self.remaining, 0): # self.steps - 1 technical ending
            
            action = 0
            reward = -10
            
            self.execution_prices[self.step] = self.prices[self.step]
            self.strategy_costs[self.step] = reward

            self.prices[self.step] = self.datahandler.data[self.step]
            
            state = np.array([
                #    -1,
                    -1,
                    -1,
                    -1
                ])
            info = self.prices
            self.step += 1
            if self.step >= self.steps:
                done = True
            else:
                done = False
            return reward, state, info, done
        # Do a trade
        
        self.actions[self.step] = action #if self.step < self.steps - 2 else self.remaining
        self.inventory += action
        self.remaining = self.X_0 - self.inventory
        # Form a new price after trade

        if self.step == 0:
            self.prices[0] = self.datahandler.data[0]
        else:
            sum_ =  np.sum([self.actions[i]*self.kappa*np.exp(-self.rho*self.taus[self.step]*(self.step - i)) for i in range(self.step)])
            self.prices[self.step] = self.datahandler.data[self.step] + self.lambda_*np.sum(self.actions[:self.step-1]) + self.spread/2 + sum_

        # Receive a feedback on trade (based on previous price)
        R = 10
        self.execution_prices[self.step] = self.prices[self.step]+action/(2*self.q)
        reward = np.clip((self.prices[self.step] - self.execution_prices[self.step])/self.prices[self.step], -R, R)
        eps_penalty =  1e-6
        if action < 1:
            reward += np.log2(abs(action) + eps_penalty)
        self.rewards[self.step] = reward


        if self.step == 0:
             state = np.array([
                #0.0,
                np.log(self.prices[0]),
                self.remaining/self.X_0,
                (self.N - self.step)/self.N
                ])
        elif self.step == 1:
            state = np.array([
                #np.log(self.prices[0]),
                np.log(self.prices[1]) - np.log(self.prices[0]),
                self.remaining/self.X_0,
                (self.N - self.step)/self.N
                ])
        else:
            state = np.array([
                #np.log(self.prices[self.step-1]) - np.log(self.prices[self.step-2]),
                np.log(self.prices[self.step]) - np.log(self.prices[self.step-1]),
                self.remaining/self.X_0,
                (self.N - self.step)/self.N
                ])
            
        self.step += 1
        info = None
        if self.step >= self.steps:
            done = True
        else:
            done = False
        return reward, state, info, done


    def take_step(self, action):
        if self.verify_action(action):
            reward, state, info, done = self.take_verified_step(action)
        else:
            done = True
            reward = -self.prices[self.step]*self.X_0*2
            info = None
            state = np.array([
                #-1,
                -1,
                -1,
                -1
                ])
            self.rewards[self.step] = reward
        return reward, state, info, done
            




        





'''

        if self.step == self.steps-1 and self.remaining > 10.0: # self.steps - 1 technical ending
            
            action = 0
            reward = -1000*self.remaining/self.X_0
            
            self.execution_prices[self.step] = self.prices[self.step]
            self.rewards[self.step] = reward

            self.prices[self.step] = self.datahandler.data[self.step]
            
            state = np.array([
                #    -1,
                    -1,
                    -1,
                    -1
                ])
            info = self.prices
            self.step += 1
            if self.step >= self.steps:
                done = True
            else:
                done = False
            return reward, state, info, done


'''