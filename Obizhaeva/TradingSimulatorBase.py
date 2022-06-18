import numpy as np
from math import isclose
import gym

from rsa import verify
class TradingSimulatorBase:

    def __init__(self, env_settings, DataHandler = None):
        ## env settings
        #self.datahandler = datahandler
       
        self.datahandler = DataHandler
        self.env_settings = env_settings

        self.inventory = 1

        ## dynamics settings
        self.Nsims = self.env_settings['Nsims']  # Number of simulations
        self.Ndt = self.env_settings['Ndt']
        self.steps = self.env_settings['Ndt']
        
        self.T = 1  # Expiry
        #int(6.5*360)  # Number of time increments
        self.dt = self.T/self.Ndt  # Time change increment
        self.t = np.arange(0, self.T+0.00000001, self.dt)  # Time increments vector 

        self.dynamics_params = self.env_settings['dynamics_params'] # dict
        self.k = self.dynamics_params['k'] #0.001  # Temporary Market Impact
        self.b =self.dynamics_params['b'] #0.0001  # Permanent Price Impact Factor

        self.lam = self.dynamics_params['lam'] #1000 # Frequency of Arrival of Order-Flow Changes
        self.kappa = self.dynamics_params['kappa']# 10  # Rate of Order-Flow Mean-reversion
        self.eta_mean = self.dynamics_params['eta_mean']#5  # Mean Order-Flow Jump Size

        self.initial_price = 50  # Starting Fundamental Price
        self.initial_invt = 1  # Starting Inventory amount
        self.phi = self.dynamics_params['phi']#0.01  # Running penalty coefficient
        self.sigma = self.dynamics_params['vol']#0.1  # Volatilty (recall this is an artihmetic model)

        self.alpha = self.dynamics_params['alpha']#100  # Terminal penalty
        
        self.mu = np.full([self.Nsims, self.Ndt+1], np.nan)
        self.mu[:, 0] = 0
        self.dW = self.dt**0.5 * np.random.randn(self.Nsims, self.Ndt+1)
        self.generate_orderflow()

        self.reset()

        self.action_space = self.env_settings['state_space']
        self.observation_space =  self.env_settings['action_space']

    def reset(self):
        self.step = 0
        # Initializing variables for simulation base on computed strategy
        self.X = np.full([self.Nsims, self.Ndt+1], np.nan)  # Cost matrix of Strategy
        self.Q = np.full([self.Nsims, self.Ndt+1], np.nan)  # Inventory matrix

        self.S = np.full([self.Nsims, self.Ndt+1], np.nan)  # Execution Price matrix

        self.actions = np.full([self.Nsims, self.Ndt+1], np.nan)  # Rate of Trading matrix

        self.rewards = np.full([self.Nsims, self.Ndt+1], np.nan)
        
        # Initial  conditions
        initial_price = 50  # Starting Fundamental Price
        initial_invt = 1  # Starting Inventory amount
        self.Q[:, 0] = initial_invt
        self.S[:, 0] = initial_price  
        self.X[:, 0] = 0


        
        if self.env_settings['stochastic_reset']:
            self.mu = np.full([self.Nsims, self.Ndt+1], np.nan)  # Order Flow matrix
            self.mu[:, 0] = 0
            #self.dW = self.dt**0.5 * np.random.randn(self.Nsims, self.Ndt+1)
            self.generate_orderflow()

        return np.array([
                        np.array([0]),
                        np.log(self.S[:, 0]),
                        self.Q[:, 0],
                        np.array([1])
                        ])

    def generate_orderflow(self):
        if self.env_settings['misspecify']:
            self.kappa_false = self.dynamics_params['kappa_false']
            self.mu_false = np.full([self.Nsims, self.Ndt+1], np.nan)  # Order Flow matrix
            self.mu_false[:, 0] = 0
        for i in range(self.Ndt):
            ##  simulate order-flow forward
            # decide if an order-flow update arrives
            dn = (np.random.rand(self.Nsims, 1) < 1 - np.exp(-2 * self.lam * self.dt)).astype(int)
            # decide if it adds to the buy/sell pressure
            buysell = (np.random.rand(self.Nsims, 1) < 0.5)
            # generate the size of the order-flow impact
            eta = -self.eta_mean * np.log(np.random.rand(self.Nsims, 1))
            
            # simulate the SDE for mu forward
            if self.env_settings['misspecify']:
                self.mu_false[:, i + 1] = self.mu_false[:, i]*np.exp(-self.kappa_false * self.dt) + (eta * dn * (2 * buysell - 1)).reshape(self.Nsims)
            self.mu[:, i + 1] = self.mu[:, i]*np.exp(-self.kappa * self.dt) + (eta * dn * (2 * buysell - 1)).reshape(self.Nsims)
    
    def verify_action(self, action):
        if action < 0:
            return False
        if self.Q[0, self.step] > 1:
            return False
        return action


    #@staticmethod
    #def criterion_function(S, X, Q, alpha, phi):
    #    return X[:, -1] + Q[:, -1] * (S[:, -1]  - alpha * Q[:, -1]) - phi * np.sum(Q[:, :]**2, axis = 1)

    def take_step_(self, action):
        # ovberse S_t, X_t, Q_t, mu_t -> choose nu_t -> observe S_{t+1}, X_{t+1}, Q_{t+1}, mu_{t+1} -> get reward_{t}
        self.actions[:, self.step] = action/self.dt
        self.X[:, self.step + 1] = self.X[:, self.step] + (self.S[:, self.step] - self.k * self.actions[:, self.step]) * self.actions[:, self.step] * self.dt
        self.Q[:, self.step + 1] = self.Q[:, self.step] - self.actions[:, self.step] * self.dt
        self.S[:, self.step + 1] = self.S[:, self.step] + self.b * (self.mu[:, self.step] - self.actions[:, self.step]) * self.dt + (self.sigma * self.dW[:, self.step]).reshape(self.Nsims)
        reward = (self.X[:, self.step+1] - self.X[:, self.step]) - self.phi * self.Q[:, self.step+1]**2

        if self.step == self.steps:
            reward = self.Q[:, self.step] * (self.S[: self.step] - self.alpha * self.Q[:, self.step])

        self.rewards[:, self.step] = reward 

        if self.env_settings['misspecify']:
            state = np.array([
                #np.log(self.S[:, self.step]),
                np.log(self.S[:, self.step]),
                np.log(self.S[:, self.step]) - np.log(self.S[:, self.step-1]),
                self.Q[:, self.step]/self.Q[:, 0],
                [(self.Ndt - self.step)/self.Ndt]
                ])
        else:
            if self.step == 0:
                state = np.array([
                #np.log(self.S[:, self.step]),
                np.log(self.S[:, self.step+1]) - np.log(self.S[:, self.step]),
                np.log(self.S[:, self.step+1]) - np.log(self.S[:, self.step]),
                self.Q[:, self.step+1]/self.Q[:, 0],
                [(self.Ndt - self.step-1)/self.Ndt]
                ])
            else:
                state = np.array([
                    #np.log(self.S[:, self.step]),
                np.log(self.S[:, self.step]) - np.log(self.S[:, self.step-1]),
                np.log(self.S[:, self.step+1]) - np.log(self.S[:, self.step]),
                self.Q[:, self.step+1]/self.Q[:, 0],
                [(self.Ndt - self.step-1)/self.Ndt]
                ])
    
        self.step += 1
        info = None
        if self.step >= self.steps:
            done = True
        else:
            done = False
        return reward, state, info, done
    def take_step(self, action):
        if self.step == self.steps + 1:
            done = True
            reward = [0]
            state = np.array([
                [-1],
                [-1],
                [-1],
                [-1],
                ])
            info = self.S
            self.step += 1
            if self.step >= self.steps:
                done = True
            else:
                done = False
            return reward, state, info, done
        if self.verify_action(action): # verify action
            reward, state, info, done = self.take_step_(action)
        else:
            done = True
            reward = [-1274998.955965912]####### penalty
            info = None
            state = np.array([
                [-1],
                [-1],
                [-1],
                [-1],
                ])
            self.rewards[:, self.step] = reward[0]
        return reward, state, info, done