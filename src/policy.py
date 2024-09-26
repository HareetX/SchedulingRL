import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils

class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, device, lr=3e-4, lmbda=0.95, num_epochs=10, epsilon=0.2, gamma=0.99, value_coeff=0.5, entropy_coeff=0.01, max_grad_norm=0.5):
        super(PPOPolicy, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        self.lmbda = lmbda
        self.num_epochs = num_epochs
        self.epsilon = epsilon
        self.gamma = gamma
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        
        self.device = device
        self.to(self.device)
    
    def forward(self, state, valid_action_mask):
        action_probs, state_values = self.actor(state), self.critic(state)
        # if torch.isnan(action_probs).any():
        #     print("NaN values found in action_probs in forward:", action_probs)
        #     assert False
        action_probs = action_probs * valid_action_mask
        action_probs_sum = action_probs.sum(dim=-1, keepdim=True)
        action_probs = action_probs / (action_probs_sum + 1e-8)  # Add small epsilon to avoid division by zero
        return action_probs, state_values
    
    def select_action(self, state, valid_action_mask):
        state = torch.FloatTensor(state).to(self.device)
        valid_action_mask = torch.FloatTensor(valid_action_mask).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.forward(state, valid_action_mask)
        if torch.isnan(action_probs).any():
            print("NaN values found in action_probs in select_action:", action_probs)
            assert False
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()

    def update(self, experiences):
        # Unpack experiences: states, actions, rewards, next_states, dones, valid_action_masks
        states, actions, rewards, next_states, dones, valid_action_masks = experiences

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(rewards).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).view(-1, 1).to(self.device)
        valid_action_masks = torch.FloatTensor(valid_action_masks).to(self.device)

        # Compute advantages and returns
        td_target = rewards + (1 - dones) * self.gamma * self.critic(next_states)
        td_delta = td_target - self.critic(states)
        advantages = utils.compute_advantage(td_delta.cpu(), self.gamma, self.lmbda).to(self.device)
        
        # Compute old action probabilities with mask
        old_action_probs, _ = self.forward(states, valid_action_masks)
        if torch.isnan(old_action_probs).any():
            print("NaN values found in old_action_probs in update:", old_action_probs)
            assert False
        old_log_probs = torch.log(old_action_probs.gather(1, actions)+1e-8).detach() # Add small epsilon to avoid log(0)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for multiple epochs
        total_loss = 0
        total_actor_loss = 0
        total_critic_loss = 0

        for _ in range(self.num_epochs):
            # Compute current action probabilities and values with mask
            action_probs, state_values = self.forward(states, valid_action_masks)
            if torch.isnan(action_probs).any():
                print("States in update:", states)
                print("Valid action masks in update:", valid_action_masks)
                print("NaN values found in action_probs in update:", action_probs)
                print("state_values in update:", state_values)
                assert False
            dist = torch.distributions.Categorical(action_probs)
            log_probs = torch.log(action_probs.gather(1, actions) + 1e-8)  # Add small epsilon to avoid log(0)
            
            # Compute ratio and surrogate loss
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            critic_loss = nn.MSELoss()(state_values.squeeze(), td_target.squeeze().detach())
            
            # Compute entropy bonus
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy

            # Accumulate losses
            total_loss += loss.item()
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

        # Return average losses over the epochs
        avg_loss = total_loss / self.num_epochs
        avg_actor_loss = total_actor_loss / self.num_epochs
        avg_critic_loss = total_critic_loss / self.num_epochs

        return avg_loss, avg_actor_loss, avg_critic_loss
