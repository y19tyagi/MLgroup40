import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
from gym.envs.toy_text.frozen_lake import generate_random_map

class DQN(nn.Module):

    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)   # first fully connected layer
        self.fc2 = nn.Linear(h1_nodes, h1_nodes)    # second fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions) # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Rectified linear unit (ReLU) activation
        x = F.relu(self.fc2(x))  # Rectified linear unit (ReLU) activation
        x = self.out(x)          # Calculating output
        return x

class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

class FrozenLakeDQL():

    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later.

    ACTIONS = ['L','D','R','U']     # for printing 0,1,2,3 => L(eft),D(own),R(ight),U(p)

    def __init__(self,
                 learning_rate_a=0.0001,
                 discount_factor_g=0.99,
                 network_sync_rate=10,
                 replay_memory_size=10000,
                 mini_batch_size=64):
        
        """Added a constructor so we can override these hyperparameters if we need (Optuna).
        All the original class-level defaults remain above for reference."""

        self.learning_rate_a = learning_rate_a
        self.discount_factor_g = discount_factor_g
        self.network_sync_rate = network_sync_rate
        self.replay_memory_size = replay_memory_size
        self.mini_batch_size = mini_batch_size
        self.policy_dqn = None 

    def train(self, episodes, render=False, is_slippery=False):

        num_states = 16
        num_actions = 4

        # Creating policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        
        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        self.policy_dqn = policy_dqn

        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.AdamW(policy_dqn.parameters(), lr=self.learning_rate_a)

        # This list keeps track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []

        # Creating replay memory
        memory = ReplayMemory(self.replay_memory_size)
        epsilon = 1
        step_count = 0 
            
        for i in range(episodes):

            env = gym.make('FrozenLake-v1', 
                           desc=generate_random_map(size=4), 
                           is_slippery=is_slippery,
                           render_mode='human' if render else None)
            # Resetting the environment
            state = env.reset()[0]
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions    

            # The agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while not (terminated or truncated):
                # Selecting action based on epsilon-greedy
                if random.random() < epsilon:
                    action = env.action_space.sample()  # actions: 0=left,1=down,2=right,3=up
                else:
                    with torch.no_grad():
                        action = self.policy_dqn(self.state_to_dqn_input(state, 16)).argmax().item()

                new_state, reward, terminated, truncated, _ = env.step(action)
                memory.append((state, action, new_state, reward, terminated))
                state = new_state
                step_count += 1

                # Checking if the replay memory has enough samples:
                if len(memory) > self.mini_batch_size:

                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    if step_count % 1000 == 0:  
                        epsilon = max(epsilon - 1e-5, 0.01)
                    epsilon_history.append(epsilon)

                    if step_count % self.network_sync_rate == 0:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
            env.close()

        torch.save(policy_dqn.state_dict(), "frozen_lake_dql.pt")

    def optimize(self, mini_batch, policy_dqn, target_dqn):
 
        num_states = policy_dqn.fc1.in_features
        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            with torch.no_grad():
                if terminated: 
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                    target = torch.FloatTensor([reward])
                else:
                    new_state_input = self.state_to_dqn_input(new_state, num_states)
                    target = torch.FloatTensor([reward + self.discount_factor_g * target_dqn(new_state_input).max().item()])
                    target = torch.tensor([target], dtype=torch.float32)

                state_input = self.state_to_dqn_input(state, num_states)
                target_q = target_dqn(state_input)

                # Adjusting the specific action to the target that was just calculated
                target_q[0, action] = target

            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)
            target_q_list.append(target_q)

        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    '''
    Converts a state (int) to a tensor representation.
    For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15.
    Parameters: state=1, num_states=16
    Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    '''
    def state_to_dqn_input(self, state:int, num_states:int) -> torch.Tensor:

        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1

        # Return a [1, num_states] tensor so it is batch-friendly in PyTorch
        return input_tensor.unsqueeze(0)  

def test(self, episodes, is_slippery=False):

    if self.policy_dqn is None:
        self.policy_dqn = DQN(16, 16, 4)

        # Load trained weights from saved file
        self.policy_dqn.load_state_dict(torch.load("frozen_lake_dql.pt"))

    self.policy_dqn.eval()
    success_count = 0 

    for _ in range(episodes):

        env = gym.make('FrozenLake-v1', 
                      desc=generate_random_map(size=4),  
                      is_slippery=is_slippery)

        state = env.reset()[0]
        terminated = False
        truncated = False

        while not (terminated or truncated):
            with torch.no_grad():

                state_input = self.state_to_dqn_input(state, 16)
                action = self.policy_dqn(state_input).argmax().item()

            # Execute action in environment
            state, reward, terminated, truncated, _ = env.step(action)

        # Check if episode ended with successful goal reach
        if terminated and reward == 1:
            success_count += 1

        env.close()

    success_rate = success_count / episodes
    print(f"Success rate on random maps: {success_rate * 10:.2f}%")

if __name__ == '__main__':
    frozen_lake = FrozenLakeDQL()
    is_slippery = False
    frozen_lake.train(10000, is_slippery=is_slippery)
    frozen_lake.test(100, is_slippery=is_slippery)
