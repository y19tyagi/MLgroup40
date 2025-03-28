import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from manos import frozen_lake_q_simple

# Assuming your FrozenLakeAgent class is defined above or imported.
# If the agent is saved in a module, you can import it accordingly.
# from your_agent_module import FrozenLakeAgent

def evaluate_agent_on_random_maps(agent, num_episodes=1000, num_maps=3):
    """
    Evaluate the trained agent on a number of randomly generated 8x8 maps.
    For each map, the agent plays `num_episodes` using a greedy policy.
    
    Returns:
        map_list: List of generated maps.
        win_rates: List of win rates (in %) for each map.
    """
    win_rates = []
    map_list = []
    
    for i in range(num_maps):
        # Generate a random 8x8 map
        random_map = generate_random_map(size=8, p=0.8)
        map_list.append(random_map)
        
        # Create the FrozenLake environment with the generated map
        env = gym.make("FrozenLake-v1", desc=random_map, is_slippery=False)
        wins = 0
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            while not done:
                # Select action greedily from the trained Q-table
                action = np.argmax(agent.q_table[state])
                state, reward, done, truncated, info = env.step(action)
                if done:
                    if reward == 1:
                        wins += 1
                    break
        
        win_rate = wins / num_episodes * 100
        win_rates.append(win_rate)
        env.close()
        print(f"Map {i+1} win rate: {win_rate:.2f}%")
    
    return map_list, win_rates

def plot_win_rates(win_rates):
    """
    Plot the win rates for the evaluated maps.
    """
    plt.figure(figsize=(8, 6))
    maps_labels = [f"Map {i+1}" for i in range(len(win_rates))]
    plt.bar(maps_labels, win_rates, color='blue')
    plt.title("Win Rates on Different FrozenLake Maps")
    plt.xlabel("Map")
    plt.ylabel("Win Rate (%)")
    plt.ylim(0, 100)
    
    for i, rate in enumerate(win_rates):
        plt.text(i, rate + 2, f"{rate:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig("model_test_results.png")
    plt.show()

if __name__ == "__main__":
    # Create an agent instance.
    # Make sure that this agent has been trained or load the trained Q-table.
    agent = frozen_lake_q_simple.FrozenLakeAgent()
    
    # Load the trained model (adjust the filename if needed)
    agent.load_model('frozen_lake_model.pkl')
    
    # Evaluate the agent on three different random maps with 1000 episodes each
    maps, win_rates = evaluate_agent_on_random_maps(agent, num_episodes=1000, num_maps=3)
    
    # Plot and save the results
    plot_win_rates(win_rates)
