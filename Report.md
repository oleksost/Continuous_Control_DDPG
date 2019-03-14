## Overview
For this project, I worked with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
I used the version of environment that contains 20 identical agents, each with its own copy of the environment to profit from distributed multi-agent training.

### Solving the Environment

Solving the environment requires obtaining an **average score** of +30 over 100 consecutive episodes and over all 20 agents.  

The given episodic task is considered as solved when the trained agent is able to obtain an average reward of +13 over 100 consecutive episodes.

##### Learning algorithm.
For solving the environment I used Deep Deterministic Policy Gradients (DDPG) algorithm. DDPG is an actor-critic, model free approach designed for continuous spaces. DDPG combines an actor model that learns an optimal policy (predict an action given a state), whereas critic is aimed to estimate the goodness of the predicted action,, that is, the state action value function (*Q* function).

During the training parameters of actor and critic are updated in alternating fashion while making use of experience replay strategy. In intuitive terms, the actor is trained predict actions, tht yield high *Q* value when evaluated by the critic. Since the network being updated also provides the target value, a local copies of the actor and critic models are created to predict the target values in order to avoid instability in training. The weights of the target predicting models are synchronized with the local networks to which the SGD updates are applied by the means of "soft" update strategy (see [original paper](https://arxiv.org/pdf/1509.02971.pdf) for details).  

##### Architecture.
Actor and critic feature a simple architecture containing two fully connected layers (128 nodes each) endowed with ReLU activations and Batch-Normalization layers in between. 
#### Hyperparameters

Main hyperparameters of the learning algorithm can be found in *ddpg_agent.py*, these were tuned to perform well on the given environment and include:


- NUM_UPDATES = 10 - number of batch updates per learning step
- UPDATE_EVERY = 20 - how often should the model be updated (one learning step in every *UPDATE_EVERY* steps )
- BUFFER_SIZE = int(1e6)  # replay buffer size
- BATCH_SIZE = 256  - minibatch size
- GAMMA = 0.9  - discount factor
- TAU = 1e-3  - for soft update of target parameters
- LR_ACTOR = 0.001 - learning rate of the actor
- LR_CRITIC = 0.001 - learning rate of the critic
- WEIGHT_DECAY = 0  - L2 weight decay
- SIGMA = 0.05 - sigma parameter of the noise process, regulates how much noise is added to the policy decision

For these environment, reducing the amount of noise to the policies's prediction (reducing amount of exploration) appeared to be the key to stable training

#### Results

In the training the agent is able to obtain and keep an average score of +30 (over 100 consecutive episodes) already after 12 episodes.
```
Episode 1	Average Score: 0.69
Episode 2	Average Score: 0.50
Episode 3	Average Score: 0.46
Episode 4	Average Score: 0.45
Episode 5	Average Score: 1.56
Episode 6	Average Score: 2.85
Episode 7	Average Score: 4.37
Episode 8	Average Score: 7.51
Episode 9	Average Score: 13.11
Episode 10	Average Score: 20.34
Episode 11	Average Score: 28.52
Episode 12	Average Score: 32.58
Episode 13	Average Score: 37.22
```

The following graph shows the dynamics of the average reward over 100 consecutive episodes, where the x-axis corresponds to the episode number and the y-axis to the score.
![](Rewards.pdf)

## Future Work

- Solving the more difficult Crawl environment (here the amount of exploration needed to solve the environment will be probably much higher)
- Further hyperparameter tuning could improve the learning speed, e.g. experimenting more with update frequencies etc..