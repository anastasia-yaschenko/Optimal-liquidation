import numpy as np
from math import isclose
from gym.spaces import Discrete

class TradingSimulator:

    def __init__(self, datahandler, steps):
        self.steps = steps #N+2

        ## Dynamics parameters
        self.N = steps - 2
        self.datahandler = datahandler
        self.X_0 = 100000
        self.q = 5000
        self.rho = 2.2231
        self.lambda_ = 1/(2*self.q)
        self.spread = 0.05
        self.kappa = 1/self.q - self.lambda_
        self.taus = [n*1/self.N for n in range(0,self.N+1)] #T
        

        ## Simulator parameters
        self.action_space = Discrete(101)
        self.step = 0
        self.prices = np.zeros(self.steps-1)
        self.actions = np.zeros(self.steps-1)
        self.execution_prices = np.zeros(self.steps-1)
        self.strategy_costs = np.zeros(self.steps-1)
        self.inventory = 0
        self.rewards = np.zeros(self.steps)

    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.prices.fill(0)
        self.rewards.fill(0)
        self.prices[0] = self.datahandler.data[0]
        self.strategy_costs.fill(0)
        self.inventory = 0
        self.bad_action_ = 0
        self.remaining = self.X_0 - self.inventory
        self.execution_prices.fill(0)
        return np.array([
                        0,
                        np.log(self.prices[0] + self.spread/2),
                        self.remaining/self.X_0,
                        (self.N - self.step)/self.N
                        ])
    def verify_action(self, action):
        #if action == 0.0:
        #    return False
        if self.remaining < 0:
            return False
        if action > self.remaining:
            return False
        return True


    def take_step_(self, action):

        if self.step == self.steps-2:
            action = self.remaining

        # Do a trade
        self.actions[self.step] = action
        self.inventory += action
        self.remaining = self.X_0 - self.inventory


        # Form a new price after trade
        if self.step == 0:
            self.prices[0] = self.datahandler.data[0] + self.lambda_ * action + self.spread/2
        else:
            sum_ = np.sum([self.actions[i]*self.kappa*np.exp(-self.rho*1/self.N*(self.step - i)) for i in range(self.step)])
            self.prices[self.step] = self.datahandler.data[self.step] + self.lambda_*np.sum(self.actions[:self.step-1]) + self.spread/2 + sum_

        # Receive a feedback on trade (based on previous price)

        R = 10
        self.execution_prices[self.step] = self.prices[self.step]+action/(2*self.q)
        reward = np.clip((self.prices[self.step] - self.execution_prices[self.step])/self.prices[self.step], -R, R)

        self.rewards[self.step] = reward

        if self.step == 0:
             state = np.array([
                0.0,
                np.log(self.prices[0] + + self.spread/2),
                self.remaining/self.X_0,
                (self.N - self.step)/self.N
                ])
        elif self.step == 1:
            state = np.array([
                np.log(self.prices[0] + self.spread/2),
                np.log(self.prices[1]) - np.log(self.prices[0]),
                self.remaining/self.X_0,
                (self.N - self.step)/self.N
                ])
        else:
            state = np.array([
                np.log(self.prices[self.step-1]) - np.log(self.prices[self.step-2]),
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
        if self.step == self.steps-1:
            done = True
            reward = 0
            state = np.array([
                -1,
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
        if self.verify_action(action):
            reward, state, info, done = self.take_step_(action)
        else:
            done = True
            reward = -10*max(self.N - self.step, 0)
            info = None
            state = np.array([
                -1,
                -1,
                -1,
                -1
                ])
            self.rewards[self.step] = reward
        return reward, state, info, done
            
            




        





