import os
import random
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from environment import SchedulingEnvironment
from agent import SchedulingAgent
from network_parser import count_network_layers
from policy import PPOPolicy
from utils import moving_average

def train(num_episodes=1000, max_steps=10):
    # Directories containing your accelerator and network files
    accelerator_dir = "../configs/accelerators/"
    network_dir = "../configs/networks/"

    # Get list of files
    accelerator_files = [f for f in os.listdir(accelerator_dir) if f.endswith('.cfg')]
    network_files = [f for f in os.listdir(network_dir) if f.endswith('.cfg')]
    layer_counts = {}
    for network_file in network_files:
        network_path = os.path.join(network_dir, network_file)
        layer_counts[network_file] = count_network_layers(network_path)
    # Print the accelerator and network files
    # print("\nAccelerator files:")
    # for file in accelerator_files:
    #     print(f"  - {file}")
    # print("\nNetwork files:")
    # for file in network_files:
    #     print(f"  - {file} (Layers: {layer_counts[file]})")

    # Initialize environment
    env = SchedulingEnvironment(
        accelerator_path=os.path.join(accelerator_dir, accelerator_files[0]),
        network_path=os.path.join(network_dir, network_files[0]),
        layer_idx=0,
        metric=0,
        max_steps=max_steps)
    
    # Initialize agent
    state_dim = env.state_dim
    action_dim = env.action_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PPOPolicy(state_dim, action_dim, 
                       device=device,
                       entropy_coeff=0.1)
    agent = SchedulingAgent(policy)
    
    # Training loop
    rewards = []
    episode_times = []
    start_time = time.time()
    
    losses = []
    actor_losses = []
    critic_losses = []

    for episode in range(num_episodes):
        episode_start_time = time.time()

        # Randomly select accelerator and network files
        accelerator_file = random.choice(accelerator_files)
        network_file = random.choice(network_files)
        layer_idx = random.randint(0, layer_counts[network_file] - 1)  # Adjust based on your network structure

        print(f"Selected configuration: {accelerator_file}, {network_file}, {layer_idx}")

        # Update environment with new files
        env.update_configuration(
            accelerator_path=os.path.join(accelerator_dir, accelerator_file),
            network_path=os.path.join(network_dir, network_file),
            layer_idx=layer_idx)

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
            # print(valid_action_mask.size)
            action, valid = agent.select_action(state, valid_action_mask)
            # step_start_time = time.time()
            next_state, reward, done, info = env.step(action, valid)
            # step_end_time = time.time()
            # step_duration = step_end_time - step_start_time
            # print(f"Action: {action}, Step duration: {step_duration:.6f} seconds")
            agent.store_experience(state, action, reward, next_state, done, valid_action_mask)
            
            if info['invalid_action']:
                invalid_actions += 1
            state = next_state
            total_reward += reward

            # if not valid:
            #     print(env.step_count)
            #     input("Pause, enter and continue...")
        
        # Update policy and record losses
        loss, actor_loss, critic_loss = agent.update_policy()
        # agent.decay_learning_rate()
        if episode >= 500 and (episode % 5 == 0):
            agent.decay_entropy_coeff()
        losses.append(loss)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        
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
    
    # Plot total loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Total Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Loss')
    plt.savefig('total_loss_plot.png')
    plt.close()

    # Plot total loss with moving_average
    plt.figure(figsize=(10, 5))
    plt.plot(moving_average(losses, 9))
    plt.title('Total Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Loss')
    plt.savefig('total_loss_plot_moving_average.png')
    plt.close()

    # Plot actor loss
    plt.figure(figsize=(10, 5))
    plt.plot(actor_losses)
    plt.title('Actor Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Actor Loss')
    plt.savefig('actor_loss_plot.png')
    plt.close()

    # Plot critic loss
    plt.figure(figsize=(10, 5))
    plt.plot(critic_losses)
    plt.title('Critic Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Critic Loss')
    plt.savefig('critic_loss_plot.png')
    plt.close()
    
    # Save the agent's policy
    torch.save(agent.policy.state_dict(), 'ppo_policy.pth')
    print("Agent's policy saved to ppo_policy.pth")
    
    # Print final scheduling table
    print("\nFinal Status:")
    env.print_status()
    
    # Save episode times and rewards to a file
    with open('training_log.txt', 'w') as f:
        f.write("Episode,Reward,Duration\n")
        for i, (reward, duration) in enumerate(zip(rewards, episode_times)):
            f.write(f"{i+1},{reward:.2f},{duration:.2f}\n")

def train_one_case(accelerator_path, network_path, num_episodes=1000, max_steps=10):
    # Get counts of layer
    # layer_counts = count_network_layers(network_path)

    # Initialize environment
    env = SchedulingEnvironment(
        accelerator_path=accelerator_path,
        network_path=network_path,
        layer_idx=0,
        metric=0,
        max_steps=max_steps)
    
    # Initialize agent
    state_dim = env.state_dim
    action_dim = env.action_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PPOPolicy(state_dim, action_dim, 
                       device=device)
    agent = SchedulingAgent(policy)
    
    # Training loop
    rewards = []
    episode_times = []
    start_time = time.time()
    
    losses = []
    actor_losses = []
    critic_losses = []

    for episode in range(num_episodes):
        episode_start_time = time.time()

        # Randomly select layer_idx
        # layer_idx = random.randint(0, layer_counts - 1)  # Adjust based on your network structure

        # Update environment with new files
        # env.update_configuration(
        #     accelerator_path=accelerator_path,
        #     network_path=network_path,
        #     layer_idx=layer_idx)

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
            action, valid = agent.select_action(state, valid_action_mask)
            # step_start_time = time.time()
            next_state, reward, done, info = env.step(action, valid)
            # step_end_time = time.time()
            # step_duration = step_end_time - step_start_time
            # print(f"Action: {action}, Step duration: {step_duration:.6f} seconds")
            agent.store_experience(state, action, reward, next_state, done, valid_action_mask)
            
            if info['invalid_action']:
                invalid_actions += 1
            state = next_state
            total_reward += reward
        
        # Update policy and record losses
        loss, actor_loss, critic_loss = agent.update_policy()
        # agent.decay_learning_rate()
        losses.append(loss)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        
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
    
    # Plot total loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Total Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Loss')
    plt.savefig('total_loss_plot.png')
    plt.close()

    # Plot actor loss
    plt.figure(figsize=(10, 5))
    plt.plot(actor_losses)
    plt.title('Actor Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Actor Loss')
    plt.savefig('actor_loss_plot.png')
    plt.close()

    # Plot critic loss
    plt.figure(figsize=(10, 5))
    plt.plot(critic_losses)
    plt.title('Critic Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Critic Loss')
    plt.savefig('critic_loss_plot.png')
    plt.close()
    
    # Save the agent's policy
    torch.save(agent.policy.state_dict(), 'ppo_policy.pth')
    print("Agent's policy saved to ppo_policy.pth")
    
    # Print final scheduling table
    print("\nFinal Status:")
    env.print_status()
    
    # Save episode times and rewards to a file
    with open('training_log.txt', 'w') as f:
        f.write("Episode,Reward,Duration\n")
        for i, (reward, duration) in enumerate(zip(rewards, episode_times)):
            f.write(f"{i+1},{reward:.2f},{duration:.2f}\n")

def load_agent(policy_path, state_dim, action_dim, device):
    policy = PPOPolicy(state_dim, action_dim, device=device)
    policy.load_state_dict(torch.load(policy_path))
    agent = SchedulingAgent(policy)
    return agent

def run(accelerator_path, network_path, layer_idx, max_steps=10):
    # Initialize environment
    env = SchedulingEnvironment(
        accelerator_path=accelerator_path,
        network_path=network_path,
        layer_idx=layer_idx,
        metric=0,
        max_steps=max_steps)
    
    # Initialize agent
    state_dim = env.state_dim
    action_dim = env.action_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = load_agent('ppo_policy.pth', state_dim, action_dim, device)
    # print("Agent's policy loaded from ppo_policy.pth")

    state = env.reset()
    done = False

    while not done:
        valid_action_mask = env.valid_action_mask
        action, valid = agent.select_action(state, valid_action_mask)
        # print(f"Action: {env.action_space[action]+[env.curr_row_idx+1]}")
        
        next_state, _, done, _ = env.step(action, valid)
        
        state = next_state
    
    # Print final scheduling table
    print("\nFinal Status:")
    env.print_status()

    return env.energy, env.cycles

def test():
    # Directories containing your accelerator and network files
    accelerator_v_dir = "../configs/accelerators_v/"
    network_v_dir = "../configs/networks_v/"

    # Get list of files
    accelerator_files = [f for f in os.listdir(accelerator_v_dir) if f.endswith('.cfg')]
    network_files = [f for f in os.listdir(network_v_dir) if f.endswith('.cfg')]
    layer_counts = {}
    for network_file in network_files:
        network_path = os.path.join(network_v_dir, network_file)
        layer_counts[network_file] = count_network_layers(network_path)
    for accelerator in accelerator_files:
        for network in network_files:
            total_energy = 0
            total_cycles = 0
            start_time = time.time()
            for layer_idx in range(layer_counts[network]):
                min_energy = float('inf')
                min_cycles = float('inf')
                for _ in range(10):
                    energy, cycles = run(
                        accelerator_path=os.path.join(accelerator_v_dir, accelerator),
                        network_path=os.path.join(network_v_dir, network),
                        layer_idx=layer_idx
                    )
                    min_energy = min(min_energy, energy)
                    min_cycles = min(min_cycles, cycles)
                total_energy += min_energy
                total_cycles += min_cycles
            end_time = time.time()
            total_time = (end_time - start_time) / 10
            
            # Write final table, total energy, and total cycles to a file
            accelerator_name = os.path.basename(accelerator).split('.')[0]
            network_name = os.path.basename(network).split('.')[0]
            filename = f"{accelerator_name}_{network_name}.txt"

            with open(filename, 'w') as f:
                f.write(f"Total Energy: {total_energy}\n")
                f.write(f"Total Cycles: {total_cycles}\n")
                f.write(f"Total Time: {total_time} s\n")

            print(f"Results written to {filename}")


if __name__ == "__main__":
    # To train a new agent
    # train(4000)
    # train_one_case(accelerator_path="../configs/accelerators/eyeriss.cfg",
    #                network_path="../configs/networks/resnet50.cfg",
    #                num_episodes=500)
    
    # To load an existing agent
    # run(
    #     accelerator_path="../configs/accelerators_v/eyeriss.cfg",
    #     network_path="../configs/networks_v/resnet50.cfg",
    #     layer_idx=0
    # )

    test()
