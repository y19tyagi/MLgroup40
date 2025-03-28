import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from frozen_lake_dql import DQN
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import os
from datetime import datetime

def ensure_results_dir():
    """Create a results directory if it doesn't exist"""
    if not os.path.exists('results'):
        os.makedirs('results')

def evaluate_frozen_lake(dqn_path="frozen_lake_dql.pt", episodes=100, is_slippery=True):
    """
    Tests our trained AI agent on new FrozenLake maps
    Returns performance statistics
    """
    print("Loading model...")
    # Creating the brain with the same structure used in training
    ai_brain = DQN(in_states=4, h1_nodes=16, out_actions=4)
    ai_brain.load_state_dict(torch.load(dqn_path, weights_only=True))
    ai_brain.eval()

    results = {
        'successes': [],
        'steps': [],
        'rewards': [],
        'success_steps': [],
        'fail_steps': [],
        'exploration_rates': [],
        'episode_rewards': []
    }

    # Initialize exploration rate
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01

    # Pre-calculate state tensors for all possible positions
    print("Pre-calculating state tensors...")
    state_tensors = {}
    for state in range(16):  # 4x4 grid
        row = state // 4
        col = state % 4
        goal_row = 3
        goal_col = 3
        row_diff = (goal_row - row)/3
        col_diff = (goal_col - col)/3
        state_tensors[state] = torch.tensor([[row/3, col/3, row_diff, col_diff]], dtype=torch.float32)

    # Generate all maps at once
    print("Generating maps...")
    maps = [generate_random_map(size=4) for _ in range(episodes)]

    print("Starting evaluation...")
    for episode in range(episodes):
        # Create environment with pre-generated map
        game_world = gym.make('FrozenLake-v1', 
                             desc=maps[episode],
                             is_slippery=is_slippery)
        current_position, _ = game_world.reset()
        game_over = False
        steps = 0
        total_reward = 0

        while not game_over:
            # Use pre-calculated state tensor
            state_tensor = state_tensors[current_position]

            # Ask AI for action with exploration
            with torch.no_grad():
                if np.random.random() < epsilon:
                    action = game_world.action_space.sample()
                else:
                    action = ai_brain(state_tensor).argmax().item()

            # Take action in the game world
            new_position, reward, terminated, truncated, _ = game_world.step(action)
            game_over = terminated or truncated
            
            # Update tracking
            current_position = new_position
            steps += 1
            total_reward += reward

        # Decay exploration rate
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Recording the results 
        won = 1 if reward == 1 else 0
        results['successes'].append(won)
        results['steps'].append(steps)
        results['rewards'].append(total_reward)
        results['exploration_rates'].append(epsilon)
        results['episode_rewards'].append(total_reward)
        
        if won:
            results['success_steps'].append(steps)
        else:
            results['fail_steps'].append(steps)
        
        game_world.close()

        # Print progress every 1000 episodes
        if (episode + 1) % 1000 == 0:
            current_success_rate = np.mean(results['successes'][-1000:]) * 100
            print(f"\nEpisodes {episode-999}-{episode+1}:")
            print(f"Success Rate: {current_success_rate:.1f}%")
            print(f"Average Steps: {np.mean(results['steps'][-1000:]):.1f}")
            print(f"Exploration Rate: {epsilon:.3f}")

    stats = {
        'success_rate': np.mean(results['successes']),
        'avg_steps': np.mean(results['steps']),
        'avg_reward': np.mean(results['rewards']),
        'avg_success_steps': np.mean(results['success_steps']) if results['success_steps'] else 0,
        'avg_fail_steps': np.mean(results['fail_steps']) if results['fail_steps'] else 0,
        'exploration_rates': results['exploration_rates'],
        'episode_rewards': results['episode_rewards'],
        'steps': results['steps'],
        'successes': results['successes']
    }

    print("\nFinal Test Results:")
    print(f"Success Rate: {stats['success_rate']*100:.1f}%")
    print(f"Average Steps: {stats['avg_steps']:.1f}")
    print(f"Successful Paths: {stats['avg_success_steps']:.1f} steps")
    print(f"Failed Paths: {stats['avg_fail_steps']:.1f} steps")
    
    return stats

def plot_training_metrics(stats):
    """Plot all training metrics in a single figure"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Success Rate
    success_rate = np.cumsum(stats['successes']) / np.arange(1, len(stats['successes']) + 1) * 100
    ax1.plot(success_rate, 'b-')
    ax1.set_title('Success Rate Over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Success Rate (%)')
    ax1.grid(True)
    ax1.set_ylim(0, 100)
    
    # Plot 2: Exploration Rate
    ax2.plot(stats['exploration_rates'], 'r-')
    ax2.set_title('Exploration Rate Decay')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.grid(True)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Episode Rewards
    ax3.plot(stats['episode_rewards'], 'g-')
    ax3.set_title('Episode Rewards')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Reward')
    ax3.grid(True)
    
    # Plot 4: Steps per Episode
    ax4.plot(stats['steps'], 'm-')
    ax4.set_title('Steps per Episode')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Steps')
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Create timestamp and descriptive filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    success_rate = stats['success_rate'] * 100
    filename = f'dql_training_meterics.png'
    
    # Save the plot with high quality
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return filename

if __name__ == "__main__":
    # Create results directory
    ensure_results_dir()
    
    print("Running evaluation...")
    test_results = evaluate_frozen_lake(episodes=100000)
    
    # Plot all metrics
    print("\nGenerating plots...")
    saved_file = plot_training_metrics(test_results)
    print(f"Plots saved as '{saved_file}'")