#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.autograd import backward

# Initial Weights and Biases
def final_init_w_b(low, high, size):
  return (high - low) * torch.rand(size[0], size[1]) + low 

# Mu
class Actor(nn.Module): 
  """
  input : State
  output : action
  """
  def __init__(self, obs_size, act_size):
    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(obs_size, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.output = nn.Linear(300, act_size)
    """"  
    # Initializing layer weights and biases
    self.layer_1.weight[:] = final_init_w_b(-(1/obs_size), 1/obs_size, self.layer_1.weight.shape)
    self.layer_1.bias[:] = final_init_w_b(-(1/obs_size), 1/obs_size, (1, self.layer_1.bias.shape[0]))
    
    self.layer_2.weight[:] = final_init_w_b(-(1/400), 1/400, self.layer_2.weight.shape)
    self.layer_2.bias[:] = final_init_w_b(-(1/400), 1/400, (1, self.layer_2.bias.shape[0]))
    
    self.output.weight[:] = final_init_w_b(-3e-3, 3e-3, self.output.weight.shape)
    self.output.bias[:] = final_init_w_b(-3e-3, 3e-3, (1, self.output.bias.shape[0]))
    """ 
    # Activation functions
    self.rlu = nn.ReLU()
    self.tnh = nn.Tanh()
      
  def forward(self, x):
    """Forward Calculation"""
    x = self.layer_1(x)
    x = self.rlu(x)
    x = self.layer_2(x)
    x = self.rlu(x)
    x = self.output(x)
    x = self.tnh(x)
    return x

# Q
class Critic(nn.Module):
  """
  input : state, action
  output : value
  """
  def __init__(self, obs_size, act_size):
    super(Critic, self).__init__() 
    self.layer_1 = nn.Linear(obs_size, 400)
    self.layer_2 = nn.Linear(400+act_size, 300) # Add actions from second hidden layer
    self.output = nn.Linear(300, 1)
    """ 
    # Initializing layer weights and biases
    self.layer_1.weight[:] = final_init_w_b(-(1/obs_size), 1/obs_size, self.layer_1.weight.shape)
    self.layer_1.bias[:] = final_init_w_b(-(1/obs_size), 1/obs_size, (1, self.layer_1.bias.shape[0]))
    
    self.layer_2.weight[:] = final_init_w_b(-(1/400), 1/400, self.layer_2.weight.shape)
    self.layer_2.bias[:] = final_init_w_b(-(1/400), 1/400, (1, self.layer_2.bias.shape[0]))
    
    self.output.weight[:] = final_init_w_b(-3e-3, 3e-3, self.output.weight.shape)
    self.output.bias[:] = final_init_w_b(-3e-3, 3e-3, (1, self.output.bias.shape[0]))
    """ 
    # Activation functions
    self.rlu = nn.ReLU()
    self.tnh = nn.Tanh()
  
  def forward(self, s, a):
    """Forward calculation"""

    x = self.layer_1(s)
    x = self.rlu(x)
    x = self.layer_2(torch.cat((x, a), 1)) # add actions from second hidden layer
    x = self.rlu(x)
    x = self.output(x)
    x = self.tnh(x)
    return x
