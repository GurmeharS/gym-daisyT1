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

    # import dataframe of radii
    self.data = pd.read_csv("/csv/track_1.csv")

    # General variables
    self.MAX_VEL = 30
    self.MAX_ACC = 20
    self.H = 15
    self.gas = 1000
    self.tireDur = 1000
    self.pitStop = False
    self.currVel = 0
    self.nextVel = 0
    self.finishedState = False
    self.overMaxVel = False

    self.TOTAL_STEPS = len(self.data.radius)

    for i in range(0,len(self.data.radius)):
      if self.data.radius[i] == -1:
        self.MAX_VEL_ARR[i] = self.MAX_VEL
      else:
        if np.sqrt((self.data.radius[i]) * self.H / 1000000) > 20:
            self.MAX_VEL_ARR[i] = 20
        else:
            self.MAX_VEL_ARR[i] = (np.sqrt((self.data.radius[i]) * self.H / 1000000))

    self.currStep = -1

    # Action space (what can the agent control)
    self.action_space = np.linspace(-1*self.MAX_ACC,self.MAX_ACC,2*self.MAX_ACC + 1)

    # Observation space (what can the agent see)
    self.observation_space = self.data

    # Store what agent tried
    self.curr_ep = -1
    self.memory = []

  def step(self, action):
    self.currStep += 1
    self.takeAction(action)
    curr_reward = self._get_reward()
    ob = self._get_state()
    return ob, reward, self.finishedState, {}

  def takeAction(self,action):
    
    if(action>0):
      self.gas -= 0.1*(action*action)
    elif(action<0):
      self.tireDur -= 0.1*(action*action)
    
    if(self._get_state==0):
      self.finishedState=True

    if(self.pitStop):
      action = self.MAX_ACC

    if(self.gas==0 or self.tireDur==0):
      action = self.takePitStop()

    if((action + self.currVel)>self.MAX_VEL_ARR[self.currStep]):
      self.currVel = self.MAX_VEL_ARR[self.currStep]
      self.overMaxVel = True

    if(action==0):
      dt += 1/self.currVel
    else:
      self.nextVel = np.sqrt(self.currVel*self.currVel + 2*action*1)
      dt += (self.currVel + self.nextVel)/action

    return True

  def takePitStop(self):
    self.dt += 30
    self.gas = 1000
    self.tireDur = 1000
    self.pitStop = True
    return 0

  def reset(self):
    """
    Reset environment state
    :return:
    observation object: initial observation of the space
    """
    self.currStep = -1
    self.curr_ep += 1
    self.memory.append([])
    self.gas = 1000

    return self._get_state()

  def _get_state(self):
    """Return the Observation."""
    return [self.TOTAL_STEPS - self.curr_step]

  def render(self, mode='human', close=False):
    return

  def _get_reward(self):
    if(self.pitStop):
      self.pitStop = False
      return (1/self.dt - 2)        #Defined reward
    elif(self.overMaxVel):          #Action takes us over the maximum velocity for the curve
      self.overMaxVel = False
      return (1/self.dt - 1)
    else:
      return (1/self.dt)

