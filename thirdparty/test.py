import numpy as np
from analyzer_wrapper import Analyzer

# Create an instance of the Analyzer
analyzer = Analyzer("./NeuroSpector-main/configs/accelerators/acc_2lv.cfg", 
                    "./NeuroSpector-main/configs/networks/resnet50.cfg", 
                    0)
# analyzer.print_info()
table = analyzer.get_scheduling_table()
print(table)
fixed_rows = analyzer.get_fixed_rows()
print(fixed_rows)

# Estimate the cost
analyzer.estimate_cost()
# Print the cost
total_energy = analyzer.get_total_energy()
print(f"Old Total Energy: {total_energy}")
total_cycle = analyzer.get_total_cycle()
print(f"Old Total Cycle: {total_cycle}")

# best_table = analyzer.search_optimized_table([9, 5], 2, [64, 1, 112, 112, 3, 7, 7, 1], 0)
# print(f"Best table: {np.array(best_table)}")
# best_table = analyzer.search_optimized_table_sequence([4,5,6,9], 3, table[-1], 0)
# print(f"Best table: {np.array(best_table)}")

# Estimate the cost
# analyzer.estimate_cost()
# Print the cost
# total_energy = analyzer.get_total_energy()
# print(f"New Total Energy: {total_energy}")
# total_cycle = analyzer.get_total_cycle()
# print(f"New Total Cycle: {total_cycle}")

# import time
# 
# # Record the start time for initialization
# init_start_time = time.time()
# 
# # Initialize/Update the analyzer with a scheduling table
# # analyzer.init("./NeuroSpector-main/configs/scheduling_tables/eyeriss_sample_1.cfg")
# analyzer.update_scheduling_table([[],
#                                   [],
#                                   [],
#                                   [1,1,1,1,1,1,1,1],
#                                   [1,1,1,2,1,1,1,1],
#                                   [4,1,1,1,1,1,1,1],
#                                   [1,1,1,1,1,1,1,1],
#                                   [],
#                                   [],
#                                   [16,1,56,112,3,7,7,1]])
# 
# # Record the end time for initialization and calculate the duration
# init_end_time = time.time()
# init_duration = init_end_time - init_start_time
# 
# # Record the start time for cost estimation
# estimate_start_time = time.time()
# 
# # Estimate the cost
# analyzer.estimate_cost()
# 
# # Record the end time for cost estimation and calculate the duration
# estimate_end_time = time.time()
# estimate_duration = estimate_end_time - estimate_start_time
# 
# print(f"Time to initialize: {init_duration:.6f} seconds")
# print(f"Time to estimate cost: {estimate_duration:.6f} seconds")
# 
# # Compare the durations to determine if loading the scheduling table is the bottleneck
# if init_duration > estimate_duration:
#     print("Loading the scheduling table appears to be the bottleneck.")
# else:
#     print("Loading the scheduling table does not appear to be the bottleneck.")
# 
# # Print the cost
# total_energy = analyzer.get_total_energy()
# print(f"Total Energy: {total_energy}")
# 
# total_cycle = analyzer.get_total_cycle()
# print(f"Total Cycle: {total_cycle}")