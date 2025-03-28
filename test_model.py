import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import pickle

def load_q_table(filename='frozen_lake_model.pkl'):
    """Load the trained Q-table from file"""
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
        q_table = model_data['q_table']
    print(f"Loaded Q-table with shape: {q_table.shape}")
    return q_table

def evaluate_on_map(q_table, map_desc, num_episodes=1000):
    """Evaluate the Q-table on a specific map and return win rates over time"""
    env = gym.make("FrozenLake-v1", desc=map_desc, is_slippery=True)  # Changed back to slippery
    wins = 0
    win_rates = []
    episode_rewards = []  # Track rewards per episode
    
    # Print map for debugging
    print("\nMap layout:")
    for row in map_desc:
        print(row)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        episode_reward = 0
        
        while not done and steps < 200:  # Add step limit
            # Use greedy policy (no exploration during testing)
            action = np.argmax(q_table[state])
            state, reward, done, truncated, _ = env.step(action)
            steps += 1
            episode_reward += reward
            
            if done:
                if reward == 1:
                    wins += 1
                break
        
        # Calculate win rate after each episode
        win_rate = (wins / (episode + 1)) * 100
        win_rates.append(win_rate)
        episode_rewards.append(episode_reward)
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])  # Last 100 episodes
            print(f"Episode {episode + 1}:")
            print(f"  Current win rate: {win_rate:.1f}%")
            print(f"  Average reward (last 100): {avg_reward:.3f}")
            print(f"  Steps taken: {steps}")
    
    env.close()
    return win_rates, episode_rewards

def plot_results(win_rates_list, episode_rewards_list):
    """Plot win rates and rewards over time for each map"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot win rates
    for i, win_rates in enumerate(win_rates_list):
        ax1.plot(win_rates, label=f'Map {i+1}')
    
    ax1.set_title('Win Rates Over Time for Different Maps')
    ax1.set_xlabel('Number of Episodes')
    ax1.set_ylabel('Win Rate (%)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(0, 100)
    
    # Add final win rates to the legend
    for i, win_rates in enumerate(win_rates_list):
        final_rate = win_rates[-1]
        ax1.text(len(win_rates), final_rate, f'Final: {final_rate:.1f}%', 
                verticalalignment='bottom')
    
    # Plot rewards
    for i, rewards in enumerate(episode_rewards_list):
        ax2.plot(rewards, label=f'Map {i+1}')
    
    ax2.set_title('Episode Rewards Over Time')
    ax2.set_xlabel('Number of Episodes')
    ax2.set_ylabel('Reward')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('map_test_results.png')
    plt.show()

def main():
    # Load the trained Q-table
    print("Loading trained Q-table...")
    q_table = load_q_table()
    
    # Generate 3 different random maps
    print("Generating random maps...")
    maps = [generate_random_map(size=8, p=0.8) for _ in range(3)]
    
    # Evaluate on each map
    print("Starting evaluation...")
    win_rates_list = []
    episode_rewards_list = []
    
    for i, map_desc in enumerate(maps):
        print(f"\nTesting on Map {i+1}...")
        win_rates, episode_rewards = evaluate_on_map(q_table, map_desc)
        win_rates_list.append(win_rates)
        episode_rewards_list.append(episode_rewards)
        print(f"Final win rate for Map {i+1}: {win_rates[-1]:.1f}%")
    
    # Plot results
    print("\nGenerating plot...")
    plot_results(win_rates_list, episode_rewards_list)
    print("Plot saved as 'map_test_results.png'")

if __name__ == "__main__":
    main() 