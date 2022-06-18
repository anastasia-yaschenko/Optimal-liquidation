import gym
from gym import spaces
from gym.utils import seeding
# ENV PARAMS

from DataHandler import DataHandler
from TradingSimulator import TradingSimulator


class TradingEnvironment(gym.Env):

    def __init__(self,
                data,
                trading_period): # N+1 
        self.data = data
        self.trading_period = trading_period
        self.datahandler = DataHandler(self.data, self.trading_period)
        self.simulator = TradingSimulator(self.datahandler, self.trading_period)
        #self.action_space = spaces.Discrete(3)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Returns state observation, reward, done and info"""
        #assert self.action_space.contains(action), '{} {} invalid'.format(action, type(action))
        #obs, done = self.datahandler.take_step()
        reward, state, info, done = self.simulator.take_step(action)
        
        return state, reward, info, done

    def reset(self):
        """Resets DataHandler and TradingSimulator; returns first state"""
        #self.datahandler.reset()
        #self.simulator.reset()
        return self.simulator.reset()#self.datahandler.take_step()[0]