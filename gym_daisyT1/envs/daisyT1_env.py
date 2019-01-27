import gym
import csv
import pandas as pd
import numpy as np
import math
from gym.envs.toy_text import discrete
from gym import spaces

from gym import error, spaces, utils
from gym.utils import seeding

run = 0

class DaisyT1Env(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.__version__ = "0.1.0"

    # import dataframe of radii
    self.data = pd.read_csv("/Users/gurmeharsandhu/PycharmProjects/gym-daisyT1/gym-daisyT1/gym_daisyT1/csv/track_1.csv")

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
    self.dt = 0
    self.firstDT = True
    self.pCount = 0
    self.initalA = True

    self.TOTAL_STEPS = len(self.data.radius)
    self.MAX_VEL_ARR = np.zeros((len(self.data), 1))
    for i in range(0,len(self.data.radius)):
      if self.data.radius[i] == -1:
        self.MAX_VEL_ARR[i] = self.MAX_VEL
      else:
        if np.sqrt((self.data.radius[i]) * self.H / 1000000) > self.MAX_VEL:
            self.MAX_VEL_ARR[i] = self.MAX_VEL
        else:
            self.MAX_VEL_ARR[i] = (np.sqrt((self.data.radius[i]) * self.H / 1000000))

    self.curr_step = -2

    # Action space (what can the agent control)
    self.action_space = np.array([])
    #linspace = np.linspace(-1*self.MAX_ACC,self.MAX_ACC,7)
    self.action_space = spaces.Discrete(2*self.MAX_ACC) #last arg 2*self.MAX_ACC + 1

    # Observation space (what can the agent see)
    self.observation_space = self.data

    # Store what agent tried
    self.curr_ep = 0
    self.memory = []

  def step(self, action):
    action -= self.MAX_ACC
    self.curr_step += 1
    self.takeAction(action)
    curr_reward = self._get_reward()
    ob = self._get_state()

    return ob, curr_reward, self.finishedState, {}

  def takeAction(self,action):
    self.curr_action = action
      #self.initalA = False
    if(action>0):
      self.gas -= 0.1*(action*action)
      #print(self.gas)
    elif(action<0):
      self.tireDur -= 0.1*(action*action)

    if(self._get_state==self.TOTAL_STEPS-1):
      self.finishedState=True

    if(self.pitStop):
      action = self.MAX_ACC
      self.pitStop = False
      self.pCount += 1


    if(self.gas<=15 or self.tireDur<=15):
      action = self.takePitStop()

    if((action + self.currVel)>self.MAX_VEL_ARR[self.curr_step]):
      self.currVel = self.MAX_VEL_ARR[self.curr_step]
      self.overMaxVel = True

    self.nextVel = np.sqrt(abs(self.currVel * self.currVel + 2 * action * 1))
    if(action==0):
      if (self.currVel == 0):
        self.dt = 0
      else:
        if(self.pitStop):
          self.dt += 1/self.currVel
        else:
          self.dt += 1/self.currVel
    else:

      if(self.pitStop):
        self.dt += (self.currVel + self.nextVel)/action
      else:
        self.dt = (self.currVel + self.nextVel)/action

    #if self.dt < 0:
    #  raise RuntimeError("dt < 0")

    global run
    run += 1
    """
    df = pd.DataFrame([[run, action]], columns=["Iteration", "Acceleration"])
    if run == 1:
        df.to_csv("output.csv")
    if run != 1:
        df.to_csv("output.csv", header=None, mode="a")
    print(action)
    """
    self.currVel = self.nextVel
    self.memory += [action]
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
    self.initalA = False
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
    self.dt = 0
    self.firstDT = True
    self.pCount = 0
    self.memory = []
    #print(self.memory)
    self.curr_step = -2
    self.curr_ep += 1
    #self.memory.append([])

    return self._get_state()

  def _get_state(self):
    """Return the Observation."""
    return self.curr_step

  def render(self, mode='human', close=False):
    return

  def _get_reward(self):
    #print(self.dt)
    if self.initalA:
      if self.curr_action <= 0:
        return -5
    if (self.currVel * self.currVel + 2 * self.curr_action * 1) < 0:
      return -5
    if(self.pitStop):
      #self.pitStop = False
      return (-1)        #Defined reward
    elif(self.overMaxVel):          #Action takes us over the maximum velocity for the curve
      self.overMaxVel = False
      return (-5)
    else:
      if (self.currVel <= 0):
        self.firstDT = False
        return -5
      return (1/self.dt)

