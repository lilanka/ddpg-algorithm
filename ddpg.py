import torch
import copy
import math
import numpy as np
import torch.nn as nn

from utils.buffer import Buffer
from utils.networks import Actor
from utils.networks import Critic

import matplotlib.pyplot as plt

def hard_update(target, source):
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(param.data)

class DDPG(object):
  """DDPG Algorithm"""

  def __init__(self, obs_size, act_size):

    """Constant Parameters"""
    self.lr_actor = 1e-4
    self.lr_critic = 1e-3
    self.w_decay = 1e-2 # L2 weight decay for Q
    self.to = 1e-3 # Soft target update
    self.buffer_size = 1e-6
    self.minibatch_size = 64
    self.mean = 0
    self.sigma = 1 
    self.gemma = 0.99

    # Initializing networks
    self.actor = Actor(obs_size, act_size)
    self.actor_bar = Actor(obs_size, act_size)

    self.critic = Critic(obs_size, act_size)
    self.critic_bar = Critic(obs_size, act_size)

    # Make actor_bar and critic_bar with same weights
    hard_update(self.actor_bar, self.actor)
    hard_update(self.critic_bar, self.critic)

    # Initializing buffer
    self.buffer = Buffer(self.buffer_size)

    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic, weight_decay=self.w_decay)
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
    self.mse_loss = nn.MSELoss()

  def exploration_policy(self, shape):
    """ 
    Gaussian noise
    """
    return torch.normal(self.mean, self.sigma, size=(shape[0], shape[1]))

  def action_taking(self, is_explore, state):
    """
    Select the actions
    """
    a = self.actor.forward(state)
    if is_explore:
      return a + self.exploration_policy(a.shape)
    return a 

  def converter(self, arr, to):
    """
    Convert a = (numpy to torch.tensor) or ~a
    """
    if to == "Torch":
      return torch.from_numpy(arr)
    return arr.detach().numpy()

  def store_data(self, data):
    """
    Store data on the buffer
    """
    self.buffer.store(data)

  def train_algorithm(self):
    """Train the algorithm"""
    # Set of mini-batch
    data = self.buffer.get(self.minibatch_size)
    """
    data[0] = one sample
    data[0][0] = s : dim (1, obs_size)
    data[0][1] = a : dim (1, act_size)
    data[0][2] = r : dim (1, 1)
    """
    for sample in data:
      action = self.actor_bar.forward(sample[3])
      y_i = sample[2] + self.gemma * self.critic_bar.forward(sample[3], action)
      q = self.critic.forward(sample[0], sample[1])
  
      with torch.autograd.set_detect_anomaly(True):
        # Critic update
        self.critic_optimizer.zero_grad()
        output = self.mse_loss(q, y_i.detach()) 
        output.backward()
        self.critic_optimizer.step()

        # Actor update
        self.actor_optimizer.zero_grad()
        act = self.actor.forward(sample[0])
        q_value = self.critic.forward(sample[0], act)
        policy_loss = q_value.mean()
        policy_loss.backward()
        self.actor_optimizer.step()
      """
      # Update the target networks
      for target_param, param in zip(self.critic_bar.parameters(), self.critic.parameters()):
        target_param.data.copy_(self.to * param.data + (1.0 - self.to) * target_param.data)

      for target_param, param in zip(self.actor_bar.parameters(), self.actor.parameters()):
        target_param.data.copy_(self.to * param.data + (1.0 - self.to) * target_param.data)
      #print(self.critic.layer_1.weight, self.critic.layer_1.bias)
      """
