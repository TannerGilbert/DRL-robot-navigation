import os
import threading
import time

import numpy as np
import rclpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from td3_rl.gazebo_env import GazeboEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class TD3:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(os.path.join(directory, f"{filename}_actor.pth")))

class TD3Tester(Node):
    def __init__(self, env):
        super().__init__('td3_tester')
        self.get_logger().info("Initializing TD3 Tester Node")

        td3_rl_path = get_package_share_directory('td3_rl')
        models_path = os.path.join(td3_rl_path, "pytorch_models")

        self.seed = self.declare_parameter("td3_params.seed", 0).value
        self.max_ep = self.declare_parameter("td3_params.max_ep", -1).value
        self.file_name = self.declare_parameter("td3_params.file_name", "TD3_Turtlebot").value
        self.environment_dim = self.declare_parameter("td3_params.environment_dim", 20).value

        robot_dim = 4
        self.env = env  # Pass the already-running GazeboEnv instance
        time.sleep(5)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        state_dim = self.environment_dim + robot_dim
        action_dim = 2

        self.network = TD3(state_dim, action_dim)
        try:
            self.network.load(self.file_name, models_path)
        except:
            self.get_logger().error("Could not load the stored model parameters")
            raise ValueError("Could not load the stored model parameters")

        self.done = False
        self.episode_timesteps = 0
        self.state = self.env.reset()

        self.run_model()

    def run_model(self):
        while rclpy.ok():
            action = self.network.get_action(np.array(self.state))
            a_in = [(action[0] + 1) / 2, action[1]]
            next_state, reward, done, target = self.env.step(a_in)

            if self.max_ep != -1:
                done = 1 if self.episode_timesteps + 1 == self.max_ep else int(done)

            if done:
                self.get_logger().info('Episode finished. Restarting environment...')
                self.state = self.env.reset()
                self.done = False
                self.episode_timesteps = 0
            else:
                self.state = next_state
                self.episode_timesteps += 1

def main(args=None):
    rclpy.init(args=args)

    # Create and spin GazeboEnv in a separate thread
    env = GazeboEnv(environment_dim=20)
    executor = MultiThreadedExecutor()
    executor.add_node(env)

    # Run the environment in a separate thread so that its subscriptions work
    env_thread = threading.Thread(target=executor.spin, daemon=True)
    env_thread.start()

    # Start TD3Tester, passing the running GazeboEnv
    tester = TD3Tester(env)
    rclpy.spin(tester)

    # Cleanup
    tester.destroy_node()
    env.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
