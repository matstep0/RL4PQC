"""
DQN agent with
  - configurable fully‑connected Q‑network (hidden_layers, bias on)
  - uniform replay buffer
  - target (frozen) network for stable TD targets
  - epsilon‑greedy / Boltzmann exploration
  
    Example starting values:
    hidden_layers      : [128, 64]
    lr                 : 3e-4
    gamma              : 0.98
    buffer_size        : 50_000
    batch_size         : 64
    min_replay_size    : 1_000      # set 0 to disable warm‑up 
    target_sync_freq   : 1_000

"""


from collections import deque
import random
from typing import List, Tuple, Dict, Any
import warnings as warning

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#Debug 
from icecream import ic
#ic.enable()
#ic.disable()


# --------------------------------------------------------------------------- #
#  Replay Buffer
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    """Fixed‑size FIFO buffer storing (s, a, r, s', done)."""

    def __init__(self, capacity: int=None):
        if capacity is None:
            raise ValueError("replay capacity is required")
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, transition: Tuple):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        #print(f"Sampled batch {batch}")
        states, actions, rewards, next_states, dones = zip(*batch)

        state_batch      = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states])
        next_state_batch = torch.stack([torch.tensor(ns, dtype=torch.float32) for ns in next_states])

        action_batch  = torch.tensor(actions,  dtype=torch.long).unsqueeze(1)
        reward_batch  = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        done_batch    = torch.tensor(dones,   dtype=torch.float32).unsqueeze(1)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

# --------------------------------------------------------------------------- #
#  Q‑Network
# --------------------------------------------------------------------------- #
class DeepQNetwork(nn.Module):
    """Feed‑forward network with arbitrary hidden layers."""

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int] , bias: bool = True):
        super().__init__()
        if hidden_layers is None:
            raise ValueError("`hidden_layers` must be provided, e.g. [128,64]")
        layers = []
        last = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(last, h, bias=bias), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, output_dim, bias=bias))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.float())


# --------------------------------------------------------------------------- #
#  DQN-Agent
# --------------------------------------------------------------------------- #
class DQNAgent:
    """Deep‑Q agent with replay buffer and target network, epsilon greedy policy."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_layers: List[int] = None,
        lr: float = None,
        gamma: float = None,
        buffer_size: int  = None,
        batch_size: int = None,
        min_replay_size: int = None,  
        target_sync_freq: int = None,
        device: str = None,
        seed: int = None,
    ):
        self.device = torch.device(device or "cpu")
        self.seed = seed  
        rng = np.random.default_rng(self.seed)
        self.rng = rng
      
        self.input_dim = input_dim
        self.output_dim = output_dim

        
        # Q‑nets
        self.online_net = DeepQNetwork(input_dim, output_dim, hidden_layers).to(self.device) #Randomnes tracking probably loss due to initialisation
        self.target_net = DeepQNetwork(input_dim, output_dim, hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Hyper‑params
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size 
        self.target_sync_freq = target_sync_freq
        self.total_updates = 0

        # Optimiser / loss
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Replay buffer
        self.buffer_size = buffer_size
        self.replay = ReplayBuffer(buffer_size)  #Randomnes tracking lost as usual.


    # ---------- acting ----------# 
    def set_epsilon(self, epsilon): self.epsilon = epsilon
    # def set_temperature(self, temperature): self.temperature = temperature  
      
    @torch.no_grad()
    def select_action(self, state, policy='deterministic'):
        q_vals = self.online_net(torch.tensor(state, device=self.device).float().unsqueeze(0)).squeeze(0)
        if policy == 'deterministic':
            return int(torch.argmax(q_vals).item())
        if policy == "epsilon_greedy":
            if self.rng.random() < self.epsilon:
                return self.rng.integers(self.output_dim)
            return int(torch.argmax(q_vals).item())
        # if policy == "boltzmann":
        raise ValueError(f"Unknown exploration type {policy}.")
    
    # -----------utility---------- #
    def store(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay.append((state, action, reward, next_state, done))
        
    def _sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
    
    @torch.no_grad()    
    def _td_target(self, reward_batch, next_state_batch, done_batch):
        """y = r + γ·max_a' Q_target(s', a') with broadcasting."""
        next_q_values = self.target_net(next_state_batch).max(dim=1, keepdim=True).values
        return reward_batch + self.gamma * next_q_values * (1 - done_batch)

    # ---------- learning ---------- #
    def update_policy(self, force_update=False):

        """Update Q‑network using a mini‑batch from replay buffer."""
        
        # 1. safety first
        if  self.replay.__len__() == 0:
            return
        if  self.replay.__len__() < self.min_replay_size and not force_update:
            warning.warn("Replay buffer not full yet, skipping update. Add more samples or force update.")
            return

        # 2. sample mini‑batch
        (state_batch,
         action_batch,
         reward_batch,
         next_state_batch,
         done_batch) = self.replay.sample(self.batch_size)

        # 3. move tensors to device
        state_batch      = state_batch.to(self.device)
        action_batch     = action_batch.to(self.device)
        reward_batch     = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch       = done_batch.to(self.device)
        ic(state_batch.shape, action_batch.shape, reward_batch.shape, next_state_batch.shape, done_batch.shape)
        # 4. compute current Q(s,a)
        all_q = self.online_net(state_batch)                      # (B, |A|)
        current_q_values = all_q.gather(1, action_batch)          # -> (B, 1)        

        # 5. compute target values
        target_q_values = self._td_target(reward_batch, next_state_batch, done_batch)


        # 6. back‑prop & optimiser step
        #Huber loss regularization and gradient clipping are potential optimization (two lines actually).
        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.total_updates += 1
        if self.total_updates % self.target_sync_freq == 0:
            self._sync_target()
    
    def save_weights(self, path: str):
        torch.save(self.online_net.state_dict(), path)
        
    def load_weights(self, path: str):
        self.online_net.load_state_dict(torch.load(path, map_location=self.device))
        self._sync_target()