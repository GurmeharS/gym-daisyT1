import gym
import csv
import pandas as pd
import numpy as np
import math

from gym import error, spaces, utils
from gym.utils import seeding

class DaisyT1Env(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.__version__ = "0.1.0"

    #import dataframe of radii
    self.data = pd.read_csv("/csv/track_1.csv")

    #General variables
    self.MAX_VEL = 20
    self.MAX_ACC = 20
    self.H = 15


    for i in len(self.data.radius):
      if self.data.radius[i] == -1:
        self.MAX_VEL_ARR[i] = self.MAX_VEL
      else:
        self.MAX_VEL_ARR[i] = np.sqrt((self.data.radius[i]) * self.H / 1000000)
    self.currStep = -1
    #Action space (what can the agent control)
    self.action_space = np.linspace(-1*self.MAX_ACC,self.MAX_ACC,2*self.MAX_ACC + 1)

    #Observation space (what can the agent see)
    self.observation_space = self.data


  def step(self, action):
    self.currStep+=1


  def reset(self):
    ...
  def render(self, mode='human', close=False):
    ...
