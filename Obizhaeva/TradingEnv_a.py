import gym
from gym import spaces
from gym.utils import seeding
# ENV PARAMS


class TradingEnvironment(gym.Env):

    def __init__(self,
                TradingSimulator,
                env_settings,
                DataHandler=None,
                ): 
        self.datahandler = DataHandler
        self.env_settings = env_settings
        self.simulator = TradingSimulator(self.env_settings)
        self.observation_space  = self.simulator.observation_space
        self.action_space = self.simulator.action_space
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        reward, state, info, done = self.simulator.take_step(action)
        
        return state, reward, info, done

    def reset(self):
        return self.simulator.reset()