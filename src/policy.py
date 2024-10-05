import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import utils

class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, device, actor_lr=3e-4, critic_lr=1e-3, lmbda=0.95, num_epochs=10, epsilon=0.2, gamma=0.99, value_coeff=0.5, entropy_coeff=0.01, max_grad_norm=0.5):
        super(PPOPolicy, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
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

        action_probs = action_probs * valid_action_mask
        action_probs_sum = action_probs.sum(dim=-1, keepdim=True)

        equal_probs = valid_action_mask / valid_action_mask.sum(dim=-1, keepdim=True)

        # Use where to choose between equal probs and normalized probs
        action_probs = torch.where(
            action_probs_sum > 0,
            action_probs / action_probs_sum,
            equal_probs
        )

        return action_probs, state_values, (action_probs_sum > 0)
    
    def select_action(self, state, valid_action_mask):
        state = torch.FloatTensor(state).to(self.device)
        valid_action_mask = torch.FloatTensor(valid_action_mask).to(self.device)
        with torch.no_grad():
            action_probs, _, valid = self.forward(state, valid_action_mask)
        
        # if torch.isnan(action_probs).any() or torch.all(action_probs == 0):
        #     print("Invalid action_probs:", action_probs)
        #     print("State:", state)
        #     print("Valid action mask:", valid_action_mask)
        #     print("NaN values found in action_probs in select_action:", action_probs)
        #     assert False
        
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), valid

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

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Compute advantages and returns
        td_target = rewards + (1 - dones) * self.gamma * self.critic(next_states)
        td_delta = td_target - self.critic(states)
        td_target = (td_target - td_target.mean()) / (td_target.std() + 1e-8)
        advantages = utils.compute_advantage(td_delta.cpu(), self.gamma, self.lmbda).to(self.device)
        
        # Compute old action probabilities with mask
        old_action_probs, _, _ = self.forward(states, valid_action_masks)
        # if torch.isnan(old_action_probs).any():
        #     print("NaN values found in old_action_probs in update:", old_action_probs)
        #     assert False
        old_log_probs = torch.log(old_action_probs.gather(1, actions)+1e-8).detach() # Add small epsilon to avoid log(0)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for multiple epochs
        total_loss = 0
        total_actor_loss = 0
        total_critic_loss = 0

        for _ in range(self.num_epochs):
            # Compute current action probabilities and values with mask
            action_probs, state_values, _ = self.forward(states, valid_action_masks)
            # if torch.isnan(action_probs).any():
            #     print("States in update:", states)
            #     print("Valid action masks in update:", valid_action_masks)
            #     print("NaN values found in action_probs in update:", action_probs)
            #     print("state_values in update:", state_values)
            #     assert False
            dist = torch.distributions.Categorical(action_probs)
            log_probs = torch.log(action_probs.gather(1, actions) + 1e-8)  # Add small epsilon to avoid log(0)
            
            # Compute ratio and surrogate loss
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            print("actor_loss in update:", actor_loss)
            
            # Compute value loss
            critic_loss = F.smooth_l1_loss(state_values.squeeze(), td_target.squeeze().detach())
            l2_lambda = 0.01
            l2_reg = sum(p.pow(2.0).sum() for p in self.critic.parameters())
            critic_loss += l2_lambda * l2_reg
            print("critic_loss in update:", critic_loss)
            
            # Compute entropy bonus
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy
            print("loss in update:", loss)

            # Accumulate losses
            total_loss += loss.item()
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

            # Optimize critic
            self.critic_optimizer.zero_grad()
            (self.value_coeff * critic_loss).backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.max_grad_norm)
            self.critic_optimizer.step()

            # Optimize actor
            self.actor_optimizer.zero_grad()
            (actor_loss - self.entropy_coeff * entropy).backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm)
            self.actor_optimizer.step()

        # Return average losses over the epochs
        avg_loss = total_loss / self.num_epochs
        avg_actor_loss = total_actor_loss / self.num_epochs
        avg_critic_loss = total_critic_loss / self.num_epochs

        return avg_loss, avg_actor_loss, avg_critic_loss

    def decay_learning_rate(self, decay_factor=0.99):
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] *= decay_factor
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] *= decay_factor
