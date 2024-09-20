import random
import numpy as np

from analyzer_wrapper import Analyzer

class SchedulingEnvironment:
    def __init__(self, accelerator_path, network_path, layer_idx, metric, max_steps):
        # Initialize scheduling table and constraints
        self.analyzer = Analyzer(accelerator_path, network_path, layer_idx)
        
        self.table = np.array(self.analyzer.get_scheduling_table())

        self.dnn_params = self.analyzer.get_layer_parameters(layer_idx)
        
        self.num_resources = 10
        
        self.fixed_rows = []

        self.curr_row_idx = self.num_resources - 1

        self.consecutive_multi_row_actions = 0

        self.metric = metric

        self.energy = 0
        self.cycles = 0
        self.initial_energy = 0
        self.initial_cycles = 0

        self.step_count = 0
        self.max_steps = max_steps

        self.action_history = [9] * 10  # Store last 10 actions
        
        self.reset()
    
    def reset(self):
        # Reset the environment to initial state
        # Initialize table with 1s and last row with DNN parameters
        self.table = np.ones((self.num_resources, len(self.dnn_params)), dtype=int)
        self.table[-1] = self.dnn_params

        self.analyzer.update_scheduling_table(self.table)
        self.analyzer.estimate_cost()

        self.energy = self.analyzer.get_total_energy()
        self.cycles = self.analyzer.get_total_cycle()
        self.initial_energy = self.energy
        self.initial_cycles = self.cycles

        # Reset other environment variables if needed
        self.fixed_rows = self.analyzer.get_fixed_rows()
        self.curr_row_idx = self.num_resources - 1

        self.consecutive_multi_row_actions = 0

        self.step_count = 0

        # Include normalized performance metrics
        normalized_energy = self.energy / self.initial_energy
        normalized_cycles = self.cycles / self.initial_cycles

        self.action_history = [9] * 10

        # Return the initial state
        return np.concatenate([
            self.table.flatten(),
            self.fixed_rows,
            [self.curr_row_idx],
            [normalized_energy, normalized_cycles],
            self.action_history
        ])
    
    def step(self, action):
        changes = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],
                   [1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],[1,9],[1,10],
                   [2,3],[2,4],[2,5],[2,6],[2,7],[2,8],[2,9],[2,10],
                   [3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[3,10],
                   [4,5],[4,6],[4,7],[4,8],[4,9],[4,10],
                   [5,6],[5,7],[5,8],[5,9],[5,10],
                   [6,7],[6,8],[6,9],[6,10],
                   [7,8],[7,9],[7,10],
                   [8,9],[8,10],
                   [9,10]]
        invalid_action_fixed_row = False
        invalid_action_repetition = False
        self.consecutive_multi_row_actions = 0  # Counter for consecutive multi-row actions

        # Update action history
        self.action_history = [action] + self.action_history[:-1]

        # Check if any of the rows in the action are fixed
        if any(self.fixed_rows[i-1] for i in changes[action]):
            invalid_action_fixed_row = True
            # Skip the optimization

            # Include normalized performance metrics
            normalized_energy = self.energy / self.initial_energy
            normalized_cycles = self.cycles / self.initial_cycles

            return np.concatenate([
                self.table.flatten(),
                self.fixed_rows,
                [self.curr_row_idx],
                [normalized_energy, normalized_cycles],
                self.action_history
            ]), -100, False, {
                'invalid_action_fixed_row': invalid_action_fixed_row,
                'invalid_action_repetition': invalid_action_repetition
            }
        
        # Apply the action to the scheduling table
        optimized_rows_idx = []
        optimized_rows = []
        if self.curr_row_idx+1 not in changes[action]:
            optimized_rows.append(self.table[self.curr_row_idx])
            optimized_rows_idx.append(self.curr_row_idx)
            for i in changes[action]:
                optimized_rows.append(self.table[i-1])
                optimized_rows_idx.append(i-1)
        else:
            invalid_action_repetition = True
            # Randomly generate a new row index different from curr_row_idx
            available_rows = list(range(self.num_resources))
            optimized_rows.append(self.table[self.curr_row_idx])
            optimized_rows_idx.append(self.curr_row_idx)
            for i in changes[action]:
                if i-1 != self.curr_row_idx:
                    optimized_rows.append(self.table[i-1])
                    optimized_rows_idx.append(i-1)
                available_rows.remove(i-1)
            new_row_idx = random.choice(available_rows)
            if self.fixed_rows[new_row_idx]:
                invalid_action_fixed_row = True
                # Skip the optimization
                
                # Include normalized performance metrics
                normalized_energy = self.energy / self.initial_energy
                normalized_cycles = self.cycles / self.initial_cycles

                return np.concatenate([
                    self.table.flatten(),
                    self.fixed_rows,
                    [self.curr_row_idx],
                    [normalized_energy, normalized_cycles],
                    self.action_history
                ]), -100, False, {
                    'invalid_action_fixed_row': invalid_action_fixed_row,
                    'invalid_action_repetition': invalid_action_repetition
                }
            optimized_rows.append(self.table[new_row_idx])
            optimized_rows_idx.append(new_row_idx)
        
        # Brute force search for the best solution
        # Calculate the products of optimized_rows in the corresponding columns
        products = []
        for col in range(len(self.dnn_params)):
            product = 1
            for row in optimized_rows:
                product *= row[col]
            products.append(product)
        
        # Print status
        # print(f"Products: {products}, Optimized row indices: {optimized_rows_idx}")

        # Search for the optimized table
        optimized_table = self.analyzer.search_optimized_table(optimized_rows_idx, len(optimized_rows_idx), products, self.metric)
        # optimized_table = self.analyzer.search_optimized_table_sequence(optimized_rows_idx, len(optimized_rows_idx), products, self.metric)
        self.table = np.array(optimized_table)

        # Debug: 
        # p_energy = self.analyzer.get_total_energy()
        # p_cycles = self.analyzer.get_total_cycle()
        # optimized_table_sequence = self.analyzer.search_optimized_table_sequence(optimized_rows_idx, len(optimized_rows_idx), products, self.metric)
        # s_energy = self.analyzer.get_total_energy()
        # s_cycles = self.analyzer.get_total_cycle()
        # if not np.array_equal(optimized_table, optimized_table_sequence):
        #     print("Optimized table is not equal to optimized table sequence.")
        #     self.table = np.array(optimized_table_sequence)
        #     if p_energy > s_energy:
        #         print(f"\tEnergy: {p_energy}, Cycles: {p_cycles}")
        #         print(f"\tSequence Energy: {s_energy}, Sequence Cycles: {s_cycles}")

        new_energy = self.analyzer.get_total_energy()
        new_cycles = self.analyzer.get_total_cycle()
        if self.metric == 0:  # Energy
            improvement = self.energy - new_energy
        else:  # Cycles
            improvement = self.cycles - new_cycles
        self.energy = new_energy
        self.cycles = new_cycles

        # Calculate reward
        if improvement > 0:
            reward = improvement
            if action < 10:  # Single row optimization
                reward *= 1.5  # Bonus for single row optimization
            self.consecutive_multi_row_actions = 0
        else:
            reward = -10
            if action >= 10:  # Multi-row optimization
                self.consecutive_multi_row_actions += 1
                if self.consecutive_multi_row_actions > 3:
                    reward -= 10 * (self.consecutive_multi_row_actions - 3)  # Increasing punishment

        if invalid_action_repetition:
            reward -= 50
        
        # Update current row index
        if len(optimized_rows_idx) > 1:
            self.curr_row_idx = random.choice([idx for idx in optimized_rows_idx if idx != optimized_rows_idx[0]])

        # Check if done (you may want to define a termination condition)
        self.step_count += 1
        done = self.step_count >= self.max_steps

        # Include normalized performance metrics
        normalized_energy = self.energy / self.initial_energy
        normalized_cycles = self.cycles / self.initial_cycles

        return np.concatenate([
            self.table.flatten(),
            self.fixed_rows,
            [self.curr_row_idx],
            [normalized_energy, normalized_cycles],
            self.action_history
        ]), reward, done, {
            'invalid_action_fixed_row': invalid_action_fixed_row,
            'invalid_action_repetition': invalid_action_repetition,
            'consecutive_multi_row_actions': self.consecutive_multi_row_actions,
            'step_count': self.step_count
        }
    
    def print_status(self):
        # Print the current status
        print(f"Step Count: {self.step_count}")
        print(f"Energy: {self.energy}")
        print(f"Cycles: {self.cycles}")
        print(f"Current Row Index: {self.curr_row_idx}")
        print(f"Consecutive Multi-Row Actions: {self.consecutive_multi_row_actions}")
        # Print the scheduling table
        print("Scheduling Table:")
        for row in self.table:
            print(row)

    @property
    def state_dim(self):
        # Return the dimension of the state space
        # table size, fixed row indicators, previous row index, normalized energy and cycles, action history
        return self.table.size + self.num_resources + 1 + 2 + 10

    @property
    def action_dim(self):
        # Return the dimension of the action space
        return self.num_resources + self.num_resources * (self.num_resources - 1) // 2