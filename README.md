I want to use Reinforcement Learning method to tackle dataflow scheduling table problem.

## 1.Problem Description
The scheduling table problem involves creating a table that satisfies various constraints. The rows indicate the resources in accelerator, and the columns indicate the DNN parameters. In our case, we're dealing with a table that has the following characteristics:

1. Column Constraints:
   - Each column represents a specific parameter in DNN.
   - The product of the values in each column must be equal to corresponding parameter in DNN.

2. Row Constraints:
   - Each row represents a resource in accelerator.
   - For buffer, the tile size that calculated from the row values must be no larger than the buffer size.
   - For computing unit, the product of the values in each row must be no larger than the computing unit's capacity.

3. Cell Contents:
   - Each cell in the table represents the mapping value from parameter to resource.

4. Initialization:
   - The table starts all values are 1, except for the last row, which is initialized to corresponding parameter in DNN.
   - The goal is to adjust the table values to minimize the energy consumption and total execution time, while satisfying all constraints.

5. Optimization Objectives:
   - Minimize the energy consumption and total execution time of DNN.

The challenge lies in finding an optimal schedule that satisfies all these constraints while maximizing efficiency and fairness. This is where Reinforcement Learning can be applied to learn and improve scheduling strategies over time.


## 2.Reinforcement Learning
Reinforcement Learning (RL) is a powerful approach for solving complex optimization problems like the dataflow scheduling table problem. In this section, we'll outline the key components of our RL-based solution:

### 2.1 State Space
The state space represents the current configuration of the scheduling table. It includes:
- The current values in each cell of the table

### 2.2 Action Space
The action space defines the possible modifications an agent can make to the scheduling table. Actions include:
- Choose 1 specific row and do bruteforce search with previous row to optimize the mapping value.
- Choose 2 specific rows and do bruteforce search with previous row to optimize the mapping value.

### 2.3 Agent
The agent is the decision-making entity in our RL system. It observes the current state, selects actions to modify the scheduling table, and learns from the outcomes of its actions.

### 2.4 Environment
The environment simulates the dataflow scheduling problem. It:
- Applies the agent's actions to the scheduling table
- Checks and ensures constraint satisfaction
- Calculates energy consumption and execution time
- Provides feedback to the agent in the form of rewards

### 2.5 Reward Function
The reward function guides the agent towards optimal solutions. It considers:
- Constraint satisfaction (high positive reward for meeting all constraints)
- Improvements in energy consumption and execution time (scaled positive rewards)
- Penalties for violating constraints or worsening performance metrics
- Penalties for heavy action but no improvement

### 2.6 Policy
We employ a policy gradient method, specifically Proximal Policy Optimization (PPO), for our RL algorithm. PPO offers:
- Stable learning through trust region policy optimization
- Good sample efficiency
- Ease of implementation and tuning

The policy network takes the state as input and outputs a probability distribution over possible actions. During training, the agent samples actions from this distribution and updates the policy based on the resulting rewards.

### 2.7 Training Process
1. Initialize the scheduling table
2. For each episode:
   a. Reset the environment
   b. While not done:
      - Observe current state
      - Select action based on policy
      - Apply action and observe new state and reward
      - Store experience (state, action, reward, new state)
   c. Update policy using collected experiences
3. Repeat until convergence or maximum episodes reached

By iteratively exploring the state space and learning from outcomes, our RL agent will develop strategies to optimize the dataflow scheduling table, balancing constraint satisfaction with performance optimization.


## 3.Experiment
