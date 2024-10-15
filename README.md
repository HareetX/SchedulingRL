# Reinforcement Learning-Based Scheduling Optimization for DNN Accelerators

## Overview
This project implements a novel **Reinforcement Learning (RL)-based framework** for optimizing dataflow scheduling in Deep Neural Network (DNN) accelerators. It focuses on improving the efficiency and energy consumption of hardware accelerators through dynamic scheduling strategies.

## Features
- Develops **Proximal Policy Optimization (PPO)** for dynamic scheduling adjustment.
- Introduces an **action masking mechanism** and **entropy regularization method** to enhance training efficiency and stability.
- Utilizes a **brute-force search** to find optimal scheduling solutions.
- Supports various DNN models including **YOLO v3**, **Inception v4**, **MobileNet v3**, and **ResNet-50**.
- Demonstrates substantial improvements in **execution cycles** and **energy efficiency**.

## Results
- Compared to cutting-edge heuristic methods such as **Timeloop**, **CoSA**, **ZigZag**, and **NeuroSpector**, achieved up to a 59.7% reduction in energy consumption and a 65.6% decrease in execution cycles.

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- Numpy
- Matplotlib

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rl-dnn-accelerator-scheduling.git
