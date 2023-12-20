from collections import namedtuple, deque
from itertools import count
import math
import random

from f110_gym.envs.base_classes import Integrator
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
from torch.utils.tensorboard import SummaryWriter

PATH = "DQN-2.pth"
writer = SummaryWriter()


with open('src/map/example_map/config_example_map.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

points = np.vstack([waypoints[:, conf.wpt_xind], waypoints[:, conf.wpt_yind]]).T
print("waypoints", points)

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
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output_layer(x)
        return x





BATCH_SIZE = 10
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

action_space = gym.spaces.Discrete(3)
#print("action_space", action_space)

n_actions = action_space.n
n_observations = 1080 + 1 #il +1 è l'angolo di sterzo 

policy_net = DQN(n_observations, n_actions).to(device)

try:
    policy_net.load_state_dict(torch.load("PATH"))
    policy_net.eval()
except:
    print("no ",PATH ,"file found")


target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())


optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(100000)
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[action_space.sample()]], device=device, dtype=torch.long)

    
    



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
    writer.add_scalar("Loss/train", loss, i_episode)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    torch.save(policy_net.state_dict(), PATH)


timestep = 0.04

env  = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=timestep, integrator=Integrator.RK4)

old_y = 0

num_episodes = 999999999


count_angle = 0.0

def map_left_right(action):
    global count_angle
    max_left = -0.5
    max_right = 0.5
    tourn_step = 0.01

    if action == 0 and count_angle > max_left:
        count_angle = round(count_angle - tourn_step, 3)
    elif action == 2 and count_angle < max_right:
        count_angle = round(count_angle + tourn_step, 3)
    elif action == 1 and count_angle != 0:
        if count_angle > 0:
            count_angle = round(count_angle - tourn_step, 3)
        else:
            count_angle = round(count_angle + tourn_step, 3)

    #print("count_angle", count_angle)

    return count_angle



for i_episode in range(num_episodes):
    print("episode", i_episode)
    total_reward = 0
    # Initialize the environment and get it's state
    #get random point form points array
    point = points[np.random.randint(0, len(points))]
    x = point[0]
    y = point[1]
    direction = random.uniform(0, 2*math.pi)
    state, reward, done, info =  env.reset(np.array([[x, y, direction]]))

    while min(state["scans"][0]) < 1:
        point = points[np.random.randint(0, len(points))]
        x = point[0]
        y = point[1]
        direction = random.uniform(0, 2*math.pi)
        state, reward, done, info =  env.reset(np.array([[x, y, direction]]))
    
    obs_more_angle = np.append(state["scans"][0], count_angle)#aggiungo come feauter l'angolo di sterzo attuale

    state = torch.tensor(obs_more_angle, dtype=torch.float32, device=device).unsqueeze(0)

    for t in count():
        action =  select_action(state)

        #map action 0 to n_actions to -1 to 1
        mapped_action = map_left_right(action.item())


        observation, reward, done, info  = env.step(np.array([[mapped_action,3]]))

        obs_more_angle = np.append(observation["scans"][0], count_angle)#aggiungo come feauter l'angolo di sterzo attuale

 
        if min(observation["scans"][0]) < 0.5:
            reward = 0.001

        # if observation["poses_y"][0] < old_y:
        #     reward =  0
        # old_y = observation["poses_y"][0]
            #print("reward", reward)


        if done:
            next_state = None
            reward = 0
            count_angle = 0.0
        else:
            next_state = torch.tensor(obs_more_angle, dtype=torch.float32, device=device).unsqueeze(0)
        
        total_reward += reward
        

        reward = torch.tensor([reward], device=device)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
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

        env.render(mode='human')

        if done:
            writer.add_scalar("Reward",total_reward, i_episode)
            writer.add_scalar("time alive", t*timestep, i_episode)
            break


