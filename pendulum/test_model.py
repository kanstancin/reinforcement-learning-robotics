from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from torch.distributions.normal import Normal
from reinforce import REINFORCE
from time import sleep

import gymnasium as gym

# Create and wrap the environment
env = gym.make("InvertedPendulum-v4", render_mode='human')
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)

model_save_path = '/Users/kanstantsin/workspace/safe-rl-robotics/pendulum/models/leave_upright.pt'

seed = 1
# set seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

obs_space_dims = env.observation_space.shape[0]
# Action-space of InvertedPendulum-v4 (1)
action_space_dims = env.action_space.shape[0]

# Reinitialize agent every seed
agent = REINFORCE(obs_space_dims, action_space_dims)
agent.net = torch.load(model_save_path)
agent.net.eval()

obs, info = wrapped_env.reset(seed=seed)

done, count = False, 0

while not done:
    action = agent.sample_action(obs)
    print(action)
    # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
    # These represent the next observation, the reward from the step,
    # if the episode is terminated, if the episode is truncated and
    # additional info from the step
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    agent.rewards.append(reward)
    # print('r', reward, count)
    count += 1

    # End the episode when either truncated or terminated is true
    #  - truncated: The episode duration reaches max number of timesteps
    #  - terminated: Any of the state space values is no longer finite.
    # done = terminated or truncated
    # sleep(1)

reward_over_episodes = []
reward_over_episodes.append(wrapped_env.return_queue[-1])
rewards_over_seeds = [reward_over_episodes]


rewards_to_plot = [[np.mean(reward) for reward in rewards] for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for InvertedPendulum-v4"
)
plt.show()