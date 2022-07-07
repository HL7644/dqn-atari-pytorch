import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
from torch.distributions.categorical import Categorical

import collections
import random

import gym

from Q_module import *

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Obs_Wrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super(Obs_Wrapper, self).__init__(env)
  
  def observation(self, observation):
    #convert to tensor
    observation=torch.FloatTensor(observation).permute(2,0,1).to(device) #3x210x160 image
    #grayscale
    to_grayscale=transforms.Grayscale()
    #110x84 downsampling
    resize=transforms.Resize(size=[110,84])
    obs=resize(observation)
    obs=resize(to_grayscale(observation))
    #84x84 crop -> remove unnecessary top part of the image
    cropped_obs=obs[:,26:,:]
    #normalize into values between 0 and 1
    cropped_obs=cropped_obs/255
    return cropped_obs

class Rew_Wrapper(gym.RewardWrapper):
  def __init__(self, env):
    super(Rew_Wrapper, self).__init__(env)
  
  def reward(self, reward):
    if reward>0:
      return 1.
    elif reward==0:
      return 0.
    else:
      return -1.

class Replay_Buffer(torch.utils.data.Dataset):
  def __init__(self, max_length):
    super(Replay_Buffer, self).__init__()
    self.ep_steps=[]
    self._max_length=max_length
    self._length=0
  
  def sample_batch(self, batch_size):
    batch=[]
    batch_idx=np.random.choice(self._length, batch_size)
    for idx in batch_idx:
      batch.append(self.ep_steps[idx])
    return batch
  
  def add_item(self, ep_step):
    if self._length>=self._max_length:
      #remove earliest element
      self.ep_steps.pop(0)
      self._length=self._length-1
    #add element
    self.ep_steps.append(ep_step)
    self._length=self._length+1
    return
  
  def __getitem__(self, idx):
    return self.ep_steps[idx]
  
  def __len__(self):
    return len(self.ep_steps)

class DQN_Agent():
  def __init__(self, env_name, gamma):
    self.env_name=env_name
    self.env=Rew_Wrapper(Obs_Wrapper(gym.make(self.env_name)))
    self.test_env=Rew_Wrapper(Obs_Wrapper(gym.make(self.env_name)))
    self.gamma=gamma
    self.a_dim=self.env.action_space.n #discrete action space
    self.a_idx_list=np.arange(0,self.a_dim,1)
    self.qm=Q_Module(a_dim=self.a_dim)
    self.target_qm=Q_Module(a_dim=self.a_dim) #module with previous step parameters
    qm_params=self.qm.extract_parameters() #start w/ same paramters
    self.target_qm.inherit_parameters(qm_params)
  
  def get_a_idx(self, pre_processed_obs, a_mode, eps):
    if a_mode=='random': #random action
      a_idx=self.env.action_space.sample()
    else: #follow e_greedy
      action_probs=[eps/self.a_dim for _ in range(self.a_dim)]
      q_values=torch.zeros(self.a_dim).to(device)
      for a_idx in range(self.a_dim):
        q_values[a_idx]=self.qm(pre_processed_obs, a_idx)
      argmax_idx=torch.argmax(q_values, dim=0)
      action_probs[argmax_idx]+=(1-eps)
      a_idx=random.choices(self.a_idx_list, action_probs)[0]
    return a_idx

class Obs_Stack():
  def __init__(self, max_length):
    self.max_length=max_length
    self.data=torch.Tensor([]).to(device)
    self.length=0
  
  def add_item(self, item):
    if self.length>=self.max_length:
      self.data=self.data[1:] #remove first data
      self.length-=1
    self.data=torch.cat((self.data, item), dim=0)
    self.length+=1
    return
  
  def stack_observation(self):
    if self.length!=self.max_length: #not full
      return "invalid"
    else: #full
      return self.data

