import gymnasium as gym
import numpy as np
from time import sleep

import matplotlib.pyplot as plt
import pickle

"""
TODO
change the grid ?
implement a reward function
"""

# agent plays 2000 episodes, choosing actions using ε-greedy exploration
# use manhatan distance?
# q-table is based on learning_rate, discount, reward

class FrozenLakeAgent:
    def __init__(self):
        # Create environment
        self.env = gym.make("FrozenLake-v1",map_name="8x8", is_slippery=True,  render_mode='human') # change grid 
        # self.env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery= False, render_mode='human')
        
        # Create table for storing rewards
        self.rewards_table = np.zeros((64, 8))  # 16 states, 4 actions || change based on grid
        
        # Simple parameters
        self.learn_rate = 0.1
        self.discount = 0.95
        self.explore_chance = 0.3
        # self.reward = 0.01
        
        # Training settings
        self.num_episodes = 2000  #1000
        self.max_steps = 100 # ***
    
    def get_action(self, state): # exploring logic
        if np.random.random() < self.explore_chance:
            return self.env.action_space.sample()
        return np.argmax(self.rewards_table[state])
    
    def learn(self):
        print("Starting training...")
        
        for episode in range(self.num_episodes):
            state = self.env.reset()[0]

            for step in range(self.max_steps):
    
                action = self.get_action(state) # choose action 
                new_state, reward, done, _, _ = self.env.step(action)
                
                # Update the rewards table
                old_value = self.rewards_table[state, action]
                next_max = np.max(self.rewards_table[new_state])
                
                # Q-learning update
                new_value = old_value + self.learn_rate * (reward + self.discount * next_max - old_value)
                self.rewards_table[state, action] = new_value
                
                state = new_state
                
                if done:
                    break
            
            # Show progress
            if (episode + 1) % 500 == 0:
                print(f"Episode {episode + 1}/{self.num_episodes}")




# testing the agent with no randomness by playing 100 games
# returns win rate in percentage
# visualing using the pre-existing env
    
    def test(self):
        print("\nTesting the agent...")
        wins = 0
        tests = 100
        
        for _ in range(tests):
            state = self.env.reset()[0]
            
            for step in range(self.max_steps):
                action = np.argmax(self.rewards_table[state])
                state, reward, done, _, _ = self.env.step(action)
                
                if done:
                    wins += reward  # reward is 1 for win, 0 for loss
                    break
        
        print(f"Win rate: {wins}/{tests} ({(wins/tests)*100:.1f}%)")
    
    def show_run(self):
        print("\nShowing a run...")
        env = gym.make("FrozenLake-v1", render_mode="human")
        state = env.reset()[0]
        
        for _ in range(self.max_steps):
            action = np.argmax(self.rewards_table[state])
            state, _, done, _, _ = env.step(action)
            # sleep(1)  # Pause to make it visible
            
            if done:
                break
        
        env.close()

# Run everything
if __name__ == "__main__":
    agent = FrozenLakeAgent()
    agent.learn()
    agent.test()
    agent.show_run()