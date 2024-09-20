import numpy as np
import torch

class SchedulingAgent:
    def __init__(self, policy):
        self.policy = policy
        self.experience_buffer = []
    
    def select_action(self, state):
        # Use policy to select an action
        return self.policy.select_action(state)
    
    def store_experience(self, state, action, reward, next_state, done):
        # Store experience in buffer
        self.experience_buffer.append((state, action, reward, next_state, done))
    
    def update_policy(self):
        # Format experiences for policy update
        if self.experience_buffer:
            states, actions, rewards, next_states, dones = zip(*self.experience_buffer)
            experiences = (
                np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones)
            )
            # Use formatted experiences to update the policy
            self.policy.update(experiences)
            self.experience_buffer = []
