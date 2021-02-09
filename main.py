import gym
import torch
import numpy as np

from ddpg import DDPG

def main():
  env = gym.make('FetchPickAndPlace-v1')
  obs = env.reset()

  action = env.action_space.sample()

  OBS_SIZE = len(np.concatenate((obs['observation'], obs['achieved_goal'], obs['desired_goal']), axis=0))
  ACT_SIZE = len(action)

  epsilon = 1
  MIN_EPSILON = 0.01
  DECAY_RATE = 0.9999

  eps = 1000
  T = 100
  train = True

  # Initializing DDPG
  ddpg_algo = DDPG(OBS_SIZE, ACT_SIZE)

  # Erase buffer data
  ddpg_algo.buffer.erase()
  l = False
  for eps_number in range(eps):
    obs = env.reset()
    s = np.concatenate((obs['observation'], obs['achieved_goal'], obs['desired_goal']), axis=0)
      
    s = s.reshape((1, -1))
    s = ddpg_algo.converter(s, "Torch")
    while True: 
      env.render()

      action = ddpg_algo.action_taking(train, s.float()) 
      action_n = ddpg_algo.converter(action[0], " ")

      obs, reward, done, info = env.step(action_n)
      
      ss = np.concatenate((obs['observation'], obs['achieved_goal'], obs['desired_goal']), axis=0)
      ss = ss.reshape((1, -1))
      ss = ddpg_algo.converter(ss, "Torch")

      # Store data in the buffer
      ddpg_algo.buffer.store((s.float(), action.float(), torch.tensor([[reward]]).float(), ss.float()))
      s[:] = ss

      # Train the algorithm
      if len(ddpg_algo.buffer.buffer) >= 64:
        ddpg_algo.train_algorithm()
      if done:
        observation = env.reset()
        break
  env.close()

if __name__ == "__main__":
  main()
