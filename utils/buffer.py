#!/usr/bin/env python
import random

class Buffer(object):
  def __init__(self, size):
    self.size = size
    self.buffer = [] 
      
  def store(self, data):
    """Store data on the buffer"""
    
    if len(self.buffer) >= self.size:
      self.buffer[-1] = data
   
    self.buffer.append(data)
  
  def get(self, b_size):
    """Get mini-batch size of data"""
    
    return random.sample(self.buffer, k=b_size)
  
  def erase(self):
    """Erase the buffer"""
    self.buffer[:] = []
