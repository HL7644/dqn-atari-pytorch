import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torchvision.datasets as dsets
import numpy as np
from torch.distributions.categorical import Categorical

import collections

import gym

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Q_Module(nn.Module):
  def __init__(self, a_dim):
    super(Q_Module, self).__init__()
    self.a_dim=a_dim
    #create #a(4) sibling layers
    self.conv1=nn.Conv2d(4,16,(8,8),stride=4).to(device)
    self.conv2=nn.Conv2d(16,32,(4,4),stride=2).to(device)

    self.fc1=nn.Linear(32*9*9, 256).to(device)
    #sibling layers for 4 actions
    self.output1=nn.Linear(256,1).to(device)
    self.output2=nn.Linear(256,1).to(device)
    self.output3=nn.Linear(256,1).to(device)
    self.output4=nn.Linear(256,1).to(device)
    self.element_init()
    self.conv_w_sizes, self.conv_b_sizes, self.fc_w_sizes, self.fc_b_sizes=self.element_init()

  def element_init(self): #initialize and get sizes of parameters
    conv_w_sizes=[]
    conv_b_sizes=[]
    fc_w_sizes=[]
    fc_b_sizes=[]
    for element in self.children():
      if isinstance(element, nn.Conv2d):
        nn.init.xavier_uniform_(element.weight)
        conv_w_sizes.append(element.weight.size())
        nn.init.zeros_(element.bias)
        conv_b_sizes.append(element.bias.size())
      elif isinstance(element, nn.Linear):
        nn.init.normal_(element.weight)
        fc_w_sizes.append(element.weight.size())
        nn.init.zeros_(element.bias)
        fc_b_sizes.append(element.bias.size())
    return conv_w_sizes, conv_b_sizes, fc_w_sizes, fc_b_sizes
  
  def extract_parameters(self):
    conv_weights=torch.Tensor([]).to(device)
    conv_biases=torch.Tensor([]).to(device)
    fc_weights=torch.Tensor([]).to(device)
    fc_biases=torch.Tensor([]).to(device)
    for element in self.children():
      if isinstance(element, nn.Conv2d):
        weight=element.weight.reshape(-1)
        conv_weights=torch.cat((conv_weights, weight), dim=0)
        bias=element.bias.reshape(-1)
        conv_biases=torch.cat((conv_biases, bias), dim=0)
      elif isinstance(element, nn.Linear):
        weight=element.weight.reshape(-1)
        fc_weights=torch.cat((fc_weights, weight), dim=0)
        bias=element.bias.reshape(-1)
        fc_biases=torch.cat((fc_biases, bias), dim=0)
    return conv_weights, conv_biases, fc_weights, fc_biases
    
  def inherit_parameters(self, parameters): #parameters: conv_weights, conv_biases, fc_weights, fc_biases
    conv_weights, conv_biases, fc_weights, fc_biases=parameters
    conv_idx=0
    conv_w_idx=0
    conv_b_idx=0
    fc_idx=0
    fc_w_idx=0
    fc_b_idx=0
    for element in self.children():
      if isinstance(element, nn.Conv2d):
        w_size=self.conv_w_sizes[conv_idx]
        w_length=1
        for size in w_size:
          w_length=w_length*size
        weight=conv_weights[conv_w_idx:conv_w_idx+w_length]
        weight=weight.reshape(w_size)
        element.weight=nn.Parameter(weight)
        conv_w_idx+=w_length

        b_size=self.conv_b_sizes[conv_idx]
        b_length=1
        for size in b_size:
          b_length=b_length*size
        bias=conv_biases[conv_b_idx:conv_b_idx+b_length]
        element.bias=nn.Parameter(bias)
        conv_b_idx+=b_length

        conv_idx+=1
      elif isinstance(element, nn.Linear):
        w_size=self.fc_w_sizes[fc_idx]
        w_length=1
        for size in w_size:
          w_length=w_length*size
        weight=fc_weights[fc_w_idx:fc_w_idx+w_length]
        weight=weight.reshape(w_size)
        element.weight=nn.Parameter(weight)
        fc_w_idx+=w_length

        b_size=self.fc_b_sizes[fc_idx]
        b_length=1
        for size in b_size:
          b_length=b_length*size
        bias=fc_biases[fc_b_idx:fc_b_idx+b_length]
        element.bias=nn.Parameter(bias)
        fc_b_idx+=b_length

        fc_idx+=1
    return
  
  def forward(self, pre_processed_obs, a_idx):
    #detect_nan(pre_processed_obs)
    #print(pre_processed_obs)
    #input: 4 stacked observations -> size 4x84x84 (grayscale image of single channel)
    relu=nn.ReLU()
    convs=nn.Sequential(self.conv1, relu, self.conv2, relu)
    features=convs(pre_processed_obs)
    feature_vector=features.reshape(-1)
    if a_idx==0:
      output=self.output1
    elif a_idx==1:
      output=self.output2
    elif a_idx==2:
      output=self.output3
    else:
      output=self.output4
    fc_layers=nn.Sequential(self.fc1, relu, output)
    action_value=fc_layers(feature_vector)
    return action_value