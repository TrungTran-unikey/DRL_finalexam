import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

        self.norm = nn.LayerNorm(128)

    
    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.norm(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Agent:
    def __init__(self, input_size, output_size, name:str='Agent'):
        self.input_size = input_size
        self.output_size = output_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = DQN(input_size, output_size).to(self.device)
        self.target_model = DQN(input_size, output_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.memory = []

        self.batch_size = 2_000
        self.memory_size = 100_000
        self.gamma = 0.99
        self.eps = 1.0
        self.eps_decay = 0.99995
        self.eps_min = 0.01
        self.steps = 0
        self.update_target = 100

        self.name = name
    
    def act(self, state):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

        if np.random.rand() < self.eps:
            return np.random.choice(self.output_size)
        
        state = torch.tensor(state, dtype=torch.float).to(self.device)

        with torch.no_grad():
            return self.model.forward(state).argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def replay(self):
        self.steps += 1

        if len(self.memory) < self.batch_size:
            return 
        
        if self.steps % self.update_target == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = (torch.tensor(states, dtype=torch.float).to(self.device) - torch.sqrt(torch.tensor(722))) \
                                                                           /torch.sqrt(torch.tensor(722))
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = (torch.tensor(next_states, dtype=torch.float).to(self.device) - torch.sqrt(torch.tensor(722))) \
                                                                                     /torch.sqrt(torch.tensor(722))
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)

        current_q = self.model.forward(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_model.forward(next_states).max(1)[0].detach()
        target_q = rewards + self.gamma * next_q * (1 - dones)
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()