import torch
import torch.nn as nn
import torch.optim as optim
import random
from models import QNetwork
import numpy as np

class DQNTrainer:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, batch_size=8, device='cpu'):
        self.policy_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.current_sim = 0
        self.epsilon = 0.01
        self.epsilon_min = 0.01
        self.epsilon_decay_start = 3000
        self.total_decay_period = 7000
        self.target_update_freq = 100

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.policy_net.out.out_features - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).squeeze()
            
            for idx in range(q_values.shape[0]):
                comp = idx // 2
                direction = "Back" if idx % 2 == 0 else "Lay"
                print(f"  Action {idx}: {direction} on Competitor {comp} → Q = {q_values[idx].item():.4f}")
                
            return torch.argmax(q_values).item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def train_step(self, replay_buffer):
        batch = replay_buffer
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.current_sim += 1
        
        if self.current_sim >= self.epsilon_decay_start:
            decay_steps = self.current_sim - self.epsilon_decay_start
            decay_ratio = decay_steps / self.total_decay_period
            self.epsilon = max(self.epsilon_min, 1.0 - decay_ratio * (1.0 - self.epsilon_min))
    
        if self.current_sim % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()
 
       