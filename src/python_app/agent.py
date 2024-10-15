import numpy as np
import torch

class SchedulingAgent:
    def __init__(self, policy):
        self.policy = policy
        self.experience_buffer = []
    
    def select_action(self, state, valid_action_mask):
        # Use policy to select an action
        return self.policy.select_action(state, valid_action_mask)
    
    def store_experience(self, state, action, reward, next_state, done, valid_action_mask):
        # Store experience in buffer
        self.experience_buffer.append((state, action, reward, next_state, done, valid_action_mask))
    
    def update_policy(self):
        # Format experiences for policy update
        if self.experience_buffer:
            states, actions, rewards, next_states, dones, valid_action_masks = zip(*self.experience_buffer)
            experiences = (
                np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones),
                np.array(valid_action_masks)
            )
            self.experience_buffer = []
            
            # Use formatted experiences to update the policy
            return self.policy.update(experiences)
        else:
            return 0, 0, 0
    
    def decay_learning_rate(self, decay_factor=0.99):
        self.policy.decay_learning_rate(decay_factor)

    def decay_entropy_coeff(self, final_entropy_coeff=0.01, decay_factor=0.995):
        self.policy.decay_entropy_coeff(final_entropy_coeff, decay_factor)
