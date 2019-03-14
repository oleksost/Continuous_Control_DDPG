from unityagents import UnityEnvironment
import numpy as np

import argparse
import torch
from collections import deque
from ddpg_agent import Agent
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser(description='Train an agent:')
parser.add_argument('--env',default='Reacher.app', type=str,required=False,help='Path to the downloaded Unity environment')
parser.add_argument('--model_pth_critic',default='checkpoint_critic.pth', type=str,required=False,help='Path to the trained critic')
parser.add_argument('--model_pth_actor',default='checkpoint_actor.pth', type=str,required=False,help='Path to the trained actor')
opt=parser.parse_args()
env = UnityEnvironment(file_name=opt.env)
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)
pretrained_dict_actor = torch.load(opt.model_pth_actor, map_location=lambda storage, location: storage)
pretrained_dict_critic = torch.load(opt.model_pth_critic, map_location=lambda storage, location: storage)
model_dict_actor = agent.actor_local.state_dict()
model_dict_critic = agent.critic_local.state_dict()

# 1. filter out unnecessary keys
pretrained_dict_actor = {k: v for k, v in pretrained_dict_actor.items() if k in model_dict_actor}
pretrained_dict_critic = {k: v for k, v in pretrained_dict_critic.items() if k in model_dict_critic}
# 2. overwrite entries in the existing state dict
model_dict_actor.update(pretrained_dict_actor)
model_dict_critic.update(pretrained_dict_critic)

# 3. load the new state dict
agent.actor_local.load_state_dict(pretrained_dict_actor)
agent.critic_local.load_state_dict(pretrained_dict_critic)
agent.actor_local.eval()
agent.critic_local.eval()

env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
while True:
    actions =  agent.act(states)                       # select an action (for each agent)
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))