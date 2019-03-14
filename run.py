
from unityagents import UnityEnvironment
import numpy as np
import torch
import sys
import argparse
from collections import deque
from ddpg_agent import Agent
import matplotlib.pyplot as plt
sys.stdout.flush()
parser=argparse.ArgumentParser(description='Train an agent:') 
parser.add_argument('--env',default='Reacher_Linux_NoVis/Reacher.x86_64', type=str,required=False,help='Path to the downloaded Unity environment')
parser.add_argument('--n_episodes',default=100, type=int, required=False,help='Path to the trained critic')
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


def ddpg(n_episodes=opt.n_episodes, max_t=1000, print_every=1):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = np.zeros(num_agents)
        for t in range(max_t):
            actions = agent.act(states, True)
            env_info = env.step(actions)[brain_name]
            next_states, rewards, dones = env_info.vector_observations, env_info.rewards, env_info.local_done
            # print(next_states.shape)
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                # print(done)
                agent.step(state, action, reward, next_state, done, t)
            states = next_states
            score += rewards
            if np.any(done):  # exit loop if episode finished
                break
        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

    return scores


scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
fig.savefig('score_clip_every_20.pdf', format='pdf', bbox_inches="tight",  dpi=300)



