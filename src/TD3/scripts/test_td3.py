#!/usr/bin/env python3
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gazebo_env import GazeboEnv
import rospy
import rospkg

rospy.init_node("td3_tester", anonymous=True)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class TD3(object):
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(os.path.join(directory, f"{filename}_actor.pth")))


rospack = rospkg.RosPack()
pkg_path = rospack.get_path("td3_rl")

models_path = os.path.join(pkg_path, "pytorch_models")

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
seed = rospy.get_param("td3_params/seed", 0)
max_ep = rospy.get_param("td3_params/max_ep", -1)
file_name = rospy.get_param("td3_params/file_name", "TD3_Turtlebot")
environment_dim = rospy.get_param("td3_params/environment_dim", 20)

# Create the testing environment
robot_dim = 4
env = GazeboEnv(environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2

# Create the network
network = TD3(state_dim, action_dim)
try:
    network.load(file_name, models_path)
except:
    raise ValueError("Could not load the stored model parameters")

done = False
episode_timesteps = 0
state = env.reset()

# Begin the testing loop
while True:
    action = network.get_action(np.array(state))

    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    if max_ep != -1: # If -1 run until crash
        done = 1 if episode_timesteps + 1 == max_ep else int(done)

    if done:
        print('Episode finished. Restarting environment...')
        state = env.reset()
        done = False
        episode_timesteps = 0
    else:
        state = next_state
        episode_timesteps += 1
