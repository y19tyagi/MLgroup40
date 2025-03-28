import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
from gym.envs.toy_text.frozen_lake import generate_random_map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
 
        self.learning_rate_a = learning_rate_a        # Learning rate (increased from 0.0001)
        self.discount_factor_g = discount_factor_g    # Discount factor
        self.network_sync_rate = network_sync_rate    # Target network sync interval
        self.replay_memory_size = replay_memory_size  # Replay buffer size
        self.mini_batch_size = mini_batch_size        # Training batch size
        self.policy_dqn = None                        # Main policy network

    def train(self, episodes, render=False, is_slippery=False):
        print("Initializing training...")
        num_states = 2  # Using normalized coordinates (row, col) instead of one-hot encoding
        num_actions = 4

        # Policy and target networks 
        policy_dqn = DQN(in_states=num_states, h1_nodes=16, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=16, out_actions=num_actions)
        target_dqn.load_state_dict(policy_dqn.state_dict())  #COmbining the two networks 
        self.policy_dqn = policy_dqn

        # AdamW optimizer with its learning rate as a hyperparameter
        self.optimizer = torch.optim.AdamW(policy_dqn.parameters(), 
                                         lr=self.learning_rate_a)
        
        print("Starting training loop...")

        memory = ReplayMemory(self.replay_memory_size)
        epsilon = 1.0         # Exploration rate
        step_count = 0        # Total steps counter
        total_rewards = 0     # Track rewards for reporting

        for episode in range(episodes):

            # Create new random map for each episode
            env = gym.make('FrozenLake-v1', 
                          desc=generate_random_map(size=4),
                          is_slippery=is_slippery,
                          render_mode='human' if render else None)
            state = env.reset()[0]
            terminated = truncated = False

            while not (terminated or truncated):

                if random.random() < epsilon:
                    action = env.action_space.sample()  
                else:
                    with torch.no_grad():

                        # Greedy action from policy network
                        state_tensor = self.state_to_dqn_input(state)
                        action = policy_dqn(state_tensor).argmax().item()
                new_state, reward, terminated, truncated, _ = env.step(action)
                memory.append((state, action, new_state, reward, terminated))
                state = new_state
                step_count += 1

                # Training step when enough experiences is aquired 
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    loss = self.optimize(mini_batch, policy_dqn, target_dqn)

                    epsilon = max(epsilon * 0.995, 0.01)

                    if step_count % self.network_sync_rate == 0:
                        print(f"Syncing target network at step {step_count}")
                        target_dqn.load_state_dict(policy_dqn.state_dict())

            total_rewards += reward
            
            if (episode+1) % 100 == 0:
                avg_reward = total_rewards / 100
                print(f"Episode {episode+1}/{episodes} | Epsilon: {epsilon:.4f} | "
                      f"Avg Reward (last 100): {avg_reward:.3f} | "
                      f"Total Steps: {step_count}")
                total_rewards = 0

            env.close()

        print(f"Training completed! Total steps: {step_count}")
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
                    # Calculating target Q-value with target network
                    new_state_input = self.state_to_dqn_input(new_state)
                    target = reward + self.discount_factor_g * \
                           target_dqn(new_state_input).max().item()
                    target = torch.tensor([target], dtype=torch.float32)

                # It gives us the current q-values and modify the target for chosen action
                state_input = self.state_to_dqn_input(state)
                target_q = target_dqn(state_input).detach().clone()  # Critical detach
                target_q[0, action] = target

            # Using the current policy network's Q-value prediction
            current_q = policy_dqn(state_input)
            
            current_q_list.append(current_q)
            target_q_list.append(target_q)

        # Calculate loss and update network
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def state_to_dqn_input(self, state: int) -> torch.Tensor:
        grid_size = 4
        row = state // grid_size
        col = state % grid_size

        # Calculate goal direction
        goal_row = 3
        goal_col = 3
        row_diff = (goal_row - row)/3
        col_diff = (goal_col - col)/3

        return torch.tensor([
            [row/3, col/3, 
            row_diff, col_diff]
        ], dtype=torch.float32)

    def test(self, episodes, is_slippery=False):

        if self.policy_dqn is None:
            self.policy_dqn = DQN(2, 16, 4)
            self.policy_dqn.load_state_dict(torch.load("frozen_lake_dql.pt"))
        self.policy_dqn.eval()  # Set to testing mode
        
        wins = 0
        for _ in range(episodes):
            env = gym.make('FrozenLake-v1', 
                          desc=generate_random_map(size=4),
                          is_slippery=is_slippery)
            state = env.reset()[0]
            done = False
            
            while not done:
                state_tensor = self.state_to_dqn_input(state)
                action = self.policy_dqn(state_tensor).argmax().item()
                state, reward, done, _, _ = env.step(action)
            
            if reward == 1:
                wins += 1
            env.close()
        
        print(f"Success rate: {(wins/episodes)*100:.1f}%")

if __name__ == '__main__':
    frozen_lake = FrozenLakeDQL()
    frozen_lake.train(5000, is_slippery=False)  
    frozen_lake.test(100, is_slippery=False)