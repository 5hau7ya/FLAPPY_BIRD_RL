# FLAPPY_BIRD_RL
An AI agent that learns to play Flappy Bird from scratch using a Dueling Double Deep Q-Network. Built entirely with Python, PyTorch, and Pygame.

# Overview
This project bridges classic game development with modern deep reinforcement learning. The agent starts with zero knowledge of the game and learns optimal flight patterns through trial, error, and an epsilon-greedy exploration strategy.

# Features
* Dueling DQN Architecture: Splits state evaluation into Value and Advantage streams for highly stable learning.
* Double Q-Learning: Prevents the overestimation of action values common in standard Q-Learning.
* Experience Replay Buffer: Stores and samples past transitions to break correlation in sequential data.
* Real-time Metrics: Live plotting of scores and moving averages using Matplotlib.
