import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()
        # Network architecture with two hidden layers
        self.fc1 = nn.Linear(in_states, h1_nodes)   # First fully connected layer
        self.fc2 = nn.Linear(h1_nodes, h1_nodes)    # Second fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions) # Output layer for Q-values

    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU activation for non-linearity
        x = F.relu(self.fc2(x))  # Second ReLU activation
        return self.out(x)       # Raw Q-values for each action

class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)  # Experience replay buffer
    
    def append(self, transition):
        self.memory.append(transition)  # Add new experience

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)  # Random batch sampling

    def __len__(self):
        return len(self.memory)

class FrozenLakeDQL():
    loss_fn = nn.MSELoss()          # Loss function for Q-value regression
    optimizer = None                # Optimizer initialized later
    ACTIONS = ['L','D','R','U']     # Action mapping for debugging

    def __init__(self, learning_rate_a=0.001, discount_factor_g=0.95,
                 network_sync_rate=1000, replay_memory_size=10000, 
                 mini_batch_size=128):
        # Hyperparameter configuration
        self.learning_rate_a = learning_rate_a      # Learning rate (increased from 0.0001)
        self.discount_factor_g = discount_factor_g  # Discount factor
        self.network_sync_rate = network_sync_rate  # Target network sync interval
        self.replay_memory_size = replay_memory_size  # Replay buffer size
        self.mini_batch_size = mini_batch_size      # Training batch size
        self.policy_dqn = None                      # Main policy network

    def train(self, episodes, render=False, is_slippery=False):
        num_states = 2  # Using normalized coordinates (row, col) instead of one-hot
        num_actions = 4

        # Initialize policy and target networks
        policy_dqn = DQN(in_states=num_states, h1_nodes=16, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=16, out_actions=num_actions)
        target_dqn.load_state_dict(policy_dqn.state_dict())  # Sync networks
        self.policy_dqn = policy_dqn

        # AdamW optimizer with learning rate
        self.optimizer = torch.optim.AdamW(policy_dqn.parameters(), 
                                         lr=self.learning_rate_a)
        
        # Training state tracking
        memory = ReplayMemory(self.replay_memory_size)
        epsilon = 1.0  # Exploration rate
        step_count = 0  # Total steps counter

        for episode in range(episodes):
            # Create new random map for each episode
            env = gym.make('FrozenLake-v1', 
                          desc=generate_random_map(size=4),
                          is_slippery=is_slippery,
                          render_mode='human' if render else None)
            state = env.reset()[0]
            terminated = truncated = False

            while not (terminated or truncated):
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = env.action_space.sample()  # Random action
                else:
                    with torch.no_grad():
                        # Greedy action from policy network
                        state_tensor = self.state_to_dqn_input(state)
                        action = policy_dqn(state_tensor).argmax().item()

                # Environment interaction
                new_state, reward, terminated, truncated, _ = env.step(action)
                memory.append((state, action, new_state, reward, terminated))
                state = new_state
                step_count += 1

                # Training step when enough experiences
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Decay exploration rate exponentially
                    epsilon = max(epsilon * 0.995, 0.01)

                    # Sync target network periodically
                    if step_count % self.network_sync_rate == 0:
                        target_dqn.load_state_dict(policy_dqn.state_dict())

            env.close()
            
            # Progress reporting
            if (episode+1) % 100 == 0:
                print(f"Episode {episode+1}/{episodes} | Epsilon: {epsilon:.4f}")

        # Save trained model
        torch.save(policy_dqn.state_dict(), "frozen_lake_dql.pt")

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:
            with torch.no_grad():  # No gradient for target calculations
                if terminated:
                    target = torch.tensor([reward], dtype=torch.float32)
                else:
                    # Calculate target Q-value with target network
                    new_state_input = self.state_to_dqn_input(new_state)
                    target = reward + self.discount_factor_g * \
                           target_dqn(new_state_input).max().item()
                    target = torch.tensor([target], dtype=torch.float32)

                # Get current Q-values and modify the target for chosen action
                state_input = self.state_to_dqn_input(state)
                target_q = target_dqn(state_input).detach().clone()  # Critical detach
                target_q[0, action] = target

            # Get policy network's Q-value prediction
            current_q = policy_dqn(state_input)
            
            current_q_list.append(current_q)
            target_q_list.append(target_q)

        # Calculate loss and update network
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def state_to_dqn_input(self, state: int) -> torch.Tensor:
        """Convert state index to normalized grid coordinates"""
        grid_size = 4  # 4x4 grid
        row = state // grid_size
        col = state % grid_size
        # Normalize coordinates to [0, 1] range
        return torch.tensor([[row/(grid_size-1), col/(grid_size-1)]], 
                          dtype=torch.float32)

    def test(self, episodes, is_slippery=False):
        """Evaluate trained agent on new random maps"""
        if self.policy_dqn is None:
            self.policy_dqn = DQN(2, 16, 4)  # Match input dimensions
            self.policy_dqn.load_state_dict(torch.load("frozen_lake_dql.pt"))
        self.policy_dqn.eval()
        
        success_count = 0
        for _ in range(episodes):
            env = gym.make('FrozenLake-v1', 
                          desc=generate_random_map(size=4),
                          is_slippery=is_slippery)
            state = env.reset()[0]
            terminated = truncated = False
            
            while not (terminated or truncated):
                with torch.no_grad():
                    state_tensor = self.state_to_dqn_input(state)
                    action = self.policy_dqn(state_tensor).argmax().item()
                state, reward, terminated, truncated, _ = env.step(action)
            
            if terminated and reward == 1:
                success_count += 1
            env.close()
        
        print(f"Success rate: {success_count/episodes*100:.2f}%")

if __name__ == '__main__':
    frozen_lake = FrozenLakeDQL()
    frozen_lake.train(5000, is_slippery=False)
    frozen_lake.test(100, is_slippery=False)