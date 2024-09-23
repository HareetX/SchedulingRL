import os
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from environment import SchedulingEnvironment
from agent import SchedulingAgent
from policy import PPOPolicy

def train():
    # Directories containing your accelerator and network files
    accelerator_dir = "../configs/accelerators/"
    network_dir = "../configs/networks/"

    # Get list of files
    accelerator_files = [f for f in os.listdir(accelerator_dir) if f.endswith('.cfg')]
    network_files = [f for f in os.listdir(network_dir) if f.endswith('.cfg')]
    
    # Initialize environment
    env = SchedulingEnvironment(
        accelerator_path=os.path.join(accelerator_dir, accelerator_files[0]),
        network_path=os.path.join(network_dir, network_files[0]),
        layer_idx=0,
        metric=0,
        max_steps=10)
    
    # Initialize agent
    state_dim = env.state_dim
    action_dim = env.action_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PPOPolicy(state_dim, action_dim, device=device)
    agent = SchedulingAgent(policy)
    
    # Training loop
    num_episodes = 1000
    rewards = []
    episode_times = []
    start_time = time.time()
    
    for episode in range(num_episodes):
        episode_start_time = time.time()
        state = env.reset()
        done = False
        total_reward = 0
        invalid_actions = 0
        
        # print("Initial state shape:", state.shape)
        # print("Initial valid_action_mask:", env.valid_action_mask)
        # if np.all(env.valid_action_mask == 0):
        #     print("All actions are invalid! Resetting environment.")
        #     continue  # Skip this episode and try again
        
        while not done:
            valid_action_mask = env.valid_action_mask
            action = agent.select_action(state, valid_action_mask)
            # step_start_time = time.time()
            next_state, reward, done, info = env.step(action)
            # step_end_time = time.time()
            # step_duration = step_end_time - step_start_time
            # print(f"Action: {action}, Step duration: {step_duration:.6f} seconds")
            agent.store_experience(state, action, reward, next_state, done, valid_action_mask)
            
            if info['invalid_action_fixed_row']:
                invalid_actions += 1
            state = next_state
            total_reward += reward
        
        # Update policy
        agent.update_policy()
        
        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        
        rewards.append(total_reward)
        episode_times.append(episode_duration)
        
        print(f"Episode {episode + 1}/{num_episodes}, Duration: {episode_duration:.2f}s, Invalid Actions: {invalid_actions}")
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Duration: {episode_duration:.2f}s, Invalid Actions: {invalid_actions}")
    
    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"Training completed in {total_training_time:.2f} seconds")
    
    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('reward_plot.png')
    plt.close()
    
    # Plot episode times
    plt.figure(figsize=(10, 5))
    plt.plot(episode_times)
    plt.title('Episode Duration')
    plt.xlabel('Episode')
    plt.ylabel('Duration (seconds)')
    plt.savefig('episode_times_plot.png')
    plt.close()
    
    # Print final scheduling table
    print("\nFinal Status:")
    env.print_status()
    
    # Save episode times and rewards to a file
    with open('training_log.txt', 'w') as f:
        f.write("Episode,Reward,Duration\n")
        for i, (reward, duration) in enumerate(zip(rewards, episode_times)):
            f.write(f"{i+1},{reward:.2f},{duration:.2f}\n")

if __name__ == "__main__":
    train()
