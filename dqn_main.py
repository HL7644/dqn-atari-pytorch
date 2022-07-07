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
from dqn_base import *

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#to use Atari games
!pip install gym[atari,accept-rom-license]==0.21.0
#to use tensorboardX
!pip install tensorboardX
from tensorboardX import SummaryWriter
writer=SummaryWriter(logdir='dqn')

class DQN():
  def __init__(self, agent):
    self.agent=agent
    #set parameters to be equal in target_qm and qm
    qm_params=self.agent.qm.extract_parameters()
    self.agent.target_qm.inherit_parameters(qm_params)
    #set replay buffer
    self.replay_buffer=Replay_Buffer(max_length=1000000)
  
  def get_value_loss(self, batch_data):
    batch_size=len(batch_data)
    value_loss=torch.FloatTensor([0]).to(device)
    for ep_step in batch_data:
      pre_obs=ep_step.pre_obs
      a_idx=ep_step.action
      Q=self.agent.qm(pre_obs, a_idx)
      #compute target using parameters from previous update step
      pre_obs_f=ep_step.pre_obs_f
      Q_values=torch.zeros(self.agent.a_dim).to(device)
      for a_idx in self.agent.a_idx_list:
        Q_values[a_idx]=self.agent.target_qm(pre_obs_f, a_idx)
      Q_max,_=torch.max(Q_values, dim=0)
      target=ep_step.reward+self.agent.gamma*(1-ep_step.termin_signal)*Q_max
      #print((target-Q)**2)
      value_loss=value_loss+(target-Q)**2
    value_loss=value_loss/batch_size
    return value_loss
  
  def test_performance(self, back_track):
    acc_rewards=[]
    a_mode='policy'
    obs_stack=Obs_Stack(back_track)
    for _ in range(10): #observe 10 episodes
      accumulated_reward=0
      obs=self.agent.test_env.reset()
      obs_stack.add_item(obs)

      for _ in range(back_track-1): #fill obs-stack
        a_idx=self.agent.env.action_space.sample()
        obs_f, reward,_,_=self.agent.test_env.step(a_idx)
        obs_stack.add_item(obs_f)
        accumulated_reward+=reward
        obs=obs_f

      while True:
        pre_processed_obs=obs_stack.stack_observation()
        a_idx=self.agent.get_a_idx(pre_processed_obs, a_mode, eps=0)
        obs_f, reward, termin_signal, _=self.agent.test_env.step(a_idx)
        accumulated_reward+=reward
        obs_stack.add_item(obs_f)
        pre_processed_obs_f=obs_stack.stack_observation()
        ep_step=Ep_Step(pre_processed_obs, a_idx, reward, pre_processed_obs_f, termin_signal)
        obs=obs_f
        if termin_signal:
          obs=self.agent.test_env.reset()
          break
      acc_rewards.append(accumulated_reward)
    avg_acc_reward=sum(acc_rewards)/len(acc_rewards)
    return avg_acc_reward
    
  def train(self, batch_size, n_epochs, steps_per_epoch, start_after, update_after, update_every, action_every, back_track, eps, v_lr):
    value_optim=optim.Adam(self.agent.qm.parameters(), lr=v_lr)

    obs=self.agent.env.reset()
    #stack #batcktrack observations
    obs_stack=Obs_Stack(back_track)
    #stack observation of enough size
    obs_stack.add_item(obs)
    for _ in range(back_track-1):
      a_idx=self.agent.env.action_space.sample()
      obs_f, _,_,_=self.agent.env.step(a_idx)
      obs_stack.add_item(obs_f)
      obs=obs_f

    a_mode='random'
    update=False
    step=1
    for epoch in range(1, n_epochs+1): #perform updates on every step (sampling and updating)
      while True:
        if step%steps_per_epoch==0:
          break
        if step>start_after:
          a_mode='policy'
        if step>update_after:
          update=True
        
        pre_processed_obs=obs_stack.stack_observation()
        if step%action_every==1:
          a_idx=self.agent.get_a_idx(pre_processed_obs, a_mode, eps)
        obs_f, reward, termin_signal, _=self.agent.env.step(a_idx)
        if termin_signal:
          obs_f=self.agent.env.reset()
        #stack next obsevation
        obs_stack.add_item(obs_f)
        pre_processed_obs_f=obs_stack.stack_observation()

        ep_step=Ep_Step(pre_processed_obs, a_idx, reward, pre_processed_obs_f, termin_signal)
        self.replay_buffer.add_item(ep_step)
        #if certain amount of data are collected
        if update and step%update_every==0:
          batch_data=self.replay_buffer.sample_batch(batch_size=batch_size)
          for u_step in range(update_every):
            value_loss=self.get_value_loss(batch_data)

            value_optim.zero_grad()
            value_loss.backward()
            value_optim.step()
            #print(value_loss)
            writer.add_scalar("Q-ftn loss", value_loss.item(), step+u_step)
          #at the end of update: set target_qm to have equal params as qm
          qm_params=self.agent.qm.extract_parameters()
          self.agent.target_qm.inherit_parameters(qm_params)
        obs=obs_f
        step=step+1
      #check performance per epoch
      acc_reward=self.test_performance(self.agent.env_name, back_track)
      print("Accumulated Reward: {:.3f}".format(acc_reward))
      writer.add_scalar("Performance", acc_reward, epoch)
    return

breakout='Breakout-v0'
agent=DQN_Agent(breakout, 0.99)
dqn=DQN(agent)
dqn.train(64, 100, 4000, 10000, 1000, 50, 4, 4, 0.1, 1e-4)