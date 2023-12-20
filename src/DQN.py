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


writer = SummaryWriter()

#viene letto un file di configurazione YAML per ottenere le impostazioni dell'ambiente
#viene caricato il file di waypoints 
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

    def sample(self, batch_size): #ritorna un campione casuale di transizioni di dimensione batch_size
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

action_space = gym.spaces.Discrete(13)
#print("action_space", action_space)

n_actions = action_space.n
n_observations = 1080

#vengono inizializzate le reti neurali (polocy e taget), l'ottimizzatore e la memoria di riproduzione
policy_net = DQN(n_observations, n_actions).to(device)
try:
    policy_net.load_state_dict(torch.load("policy_net.pth"))
    policy_net.eval()
except:
    print("no policy_net.pth file found")
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(100000)
steps_done = 0


def select_action(state):
    # global steps_done
    # sample = random.random()
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    # steps_done += 1
    # if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    # else:
    #     return torch.tensor([[action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    #Se la memoria non è abbastanza grande, non viene eseguito l'aggiornamento della rete
    if len(memory) < BATCH_SIZE:
        return
    
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    #Vengono preparti i tensori necessari per il calcolo del loss, la mschera per gli stati finali
    #e i tensori per gli stati, azioni e ricompense.
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    #Calcola Q(s_t, a) - il modello calcola Q(s_t), quindi selezioniamo le colonne delle azioni intraprese. 
    #Queste sono le azioni che sarebbero state intraprese per ogni stato batch secondo policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    #Calcola V(s_{t+1}) per tutti gli stati successivi. I valori previsti delle azioni per 
    #non_final_next_states vengono calcolati in base al "vecchio" target_net; selezionando la loro 
    #migliore ricompensa con max(1).values. Questo viene unito in base alla maschera, in modo tale da 
    #avere il valore dello stato previsto o 0 nel caso in cui lo stato fosse definitivo.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    #Calcola il valori Q previsti
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    #Calcola il loss utilizzando la Huber Loss tra i valori Q predetti e i valori target
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    writer.add_scalar("Loss/train", loss, i_episode)

    # Ottimizza il modello
    optimizer.zero_grad()

    #Viene eseguita la retropropagazione
    loss.backward()
    #Si esegue un operazione di clipping dei gradienti per evitare problemi di esplosione del gradiente
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    #L'ottimizzatore esegue un passo di aggiornamento dei pesi del modello
    optimizer.step()
    torch.save(policy_net.state_dict(), "policy_net.pth")


timestep = 0.04

env  = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=timestep, integrator=Integrator.RK4)

old_y = 0

num_episodes = 999999999
for i_episode in range(num_episodes):
    print("episode", i_episode)
    total_reward = 0
    # Initialize the environment and get it's state
    #get random point form points array
    # point = points[np.random.randint(0, len(points))]
    # x = point[0]
    # y = point[1]
    # direction = random.uniform(0, 2*math.pi)
    # state, reward, done, info =  env.reset(np.array([[x, y, direction]]))

    # while min(state["scans"][0]) < 1:
    #     point = points[np.random.randint(0, len(points))]
    #     x = point[0]
    #     y = point[1]
    #     direction = random.uniform(0, 2*math.pi)
    #     state, reward, done, info =  env.reset(np.array([[x, y, direction]]))

    state, reward, done, info =  env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))


    state = torch.tensor(state["scans"][0], dtype=torch.float32, device=device).unsqueeze(0)

    for t in count():
        # Viene selezionata un'azione utilizzando la rete neurale dell'agente
        action =  select_action(state)

        #l'azione selezionata viene mappata da un indice discreto a un valore compreso tra -1 e 1
        mapped_action = (action.item() - (n_actions-1)/2)/((n_actions-1))

        #Viene eseguita l'azione nell'ambiente e ottenute le nuove osservazioni, stato, reward e flag di terminazione
        observation, reward, done, info  = env.step(np.array([[mapped_action,3]]))

        #se il valore minimo delle scansioni è minore di 0.5, il reward è 0.001
        if min(observation["scans"][0]) < 0.5:
            reward = 0.001

        # if observation["poses_y"][0] < old_y:
        #     reward =  0
        # old_y = observation["poses_y"][0]
            #print("reward", reward)


        if done:
            next_state = None
            reward = 0
        else:
            #se non è terminato, viene creato il nuovo stato basato sulle osservazioni ottenute dall'ambiente
            next_state = torch.tensor(observation["scans"][0], dtype=torch.float32, device=device).unsqueeze(0)
        
        #La ricompensa totale accumulata
        total_reward += reward
        
        reward = torch.tensor([reward], device=device)

        # La transizione viene memorizzata nella memoria
        memory.push(state, action, next_state, reward)

        # Il modello passa al nuovo stato per continuare l'esplorazione e l'apprendimento
        state = next_state

        # Esegue l'ottimizzazione della rete neurale dell'agente
        optimize_model()

        # Viene eseguito un aggiornamente "soft" della rete target, combinando i pesi della rete target con quelli della rete principale
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        #rendering dell'ambiente
        env.render(mode='human')

        if done:
            #salva su tensorboard
            writer.add_scalar("Reward",total_reward, i_episode)
            writer.add_scalar("time alive", t*timestep, i_episode)
            break


