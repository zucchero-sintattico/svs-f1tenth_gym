from collections import namedtuple, deque
import math
import random
import F110GymBase
import yaml
from argparse import Namespace
import numpy as np
from F110GymBase import PlanningStrategy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import torch.nn.init as init



with open('src/map/example_map/config_example_map.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

# set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display

# plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x




BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
print("action_space", action_space)

n_actions = 1
n_observations = 1080

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.

            return policy_net(state)
    else:

        random_action = np.array([action_space.sample()], dtype=np.float32)
        return torch.tensor(random_action, device=device, dtype=torch.float32)




def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net

    #genera un array di 0 lungo quanto policy_net(state_batch)
    #poi mette in action_batch gli action_batch
    # print(action_batch)
    # action_batch = torch.zeros(action_batch, device=device)
    # action_batch = torch.tensor(action_batch, device=device)
    # print(action_batch)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

custom_strategy = print()
gat_first_env_state = F110GymBase.F110GymBase(map, 1, conf, custom_strategy, 0.01)
first_obs, _ , _, _ = gat_first_env_state.env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))

old_y = 0

state = torch.tensor(first_obs['scans'], device=device, dtype=torch.float32)
action = select_action(state.clone().detach())

class CustomPlanningStrategy(PlanningStrategy):
    def plan(self, observation):
        global steps_done
        global old_y
        global state   
        global action     




        if observation["lap_times"] == timestep and steps_done > 1:
            reward = -1000
            #print("collision")
            next_state = None
        else:
            reward = 0.02
            next_state = state
            #se una delle scansioni è minore di 0.5
            if min(observation["scans"][0]) < 0.5:
                reward = -100
            # if old_y > observation["poses_y"][0]:
            #     reward = -10

        old_y = observation["poses_y"][0]

        reward = torch.tensor([reward], device=device)


        memory.push(state, action.max(1).indices.view(1, 1), next_state, reward)
        
        state = next_state

        # Perform one step of the optimization (on the policy network)

        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        state = torch.tensor(observation['scans'], device=device, dtype=torch.float32)
        action = select_action(state.clone().detach())
      
        #print("action", action)
        return  np.array([[action.item(),1.5]]) 


map_name = 'example_map'
num_agents = 1
timestep = 0.02
custom_strategy = CustomPlanningStrategy()

f110_gym = F110GymBase.F110GymBase(map, num_agents, conf, custom_strategy, timestep)

f110_gym.start()