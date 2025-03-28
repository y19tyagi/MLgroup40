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

def plot_test_metrics(all_results):
    """Plot test metrics for all runs"""
    plt.figure(figsize=(15, 10))
    
    colors = ['b', 'r', 'g']
    
    # Plot 1: Win Rate Over Time
    plt.subplot(2, 2, 1)
    for i, results in enumerate(all_results):
        win_rates = np.cumsum(results['successes']) / np.arange(1, len(results['successes']) + 1) * 100
        plt.plot(win_rates, f'{colors[i]}-', label=f'Run {i+1}')
    plt.title('Win Rate Over Time')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate (%)')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Episode Rewards
    plt.subplot(2, 2, 2)
    for i, results in enumerate(all_results):
        plt.plot(results['episode_rewards'], f'{colors[i]}-', label=f'Run {i+1}')
    plt.title('Episode Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    
    # Plot 3: Steps per Episode
    plt.subplot(2, 2, 3)
    for i, results in enumerate(all_results):
        plt.plot(results['steps'], f'{colors[i]}-', label=f'Run {i+1}')
    plt.title('Steps per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.grid(True)
    plt.legend()
    
    # Plot 4: Moving Average Win Rate
    plt.subplot(2, 2, 4)
    window = 100
    for i, results in enumerate(all_results):
        win_rates = np.array(results['successes']) * 100
        moving_avg = np.convolve(win_rates, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg, f'{colors[i]}-', label=f'Run {i+1}')
    plt.title(f'Moving Average Win Rate (window={window})')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot with timestamp and win rate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    win_rate = np.mean([np.mean(r['successes']) for r in all_results]) * 100
    filename = f'results/dql_test_{win_rate:.1f}percent_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nTest results plot saved as '{filename}'")

def evaluate_frozen_lake(dqn_path="frozen_lake_dql.pt", episodes=1000, num_runs=3, is_slippery=True):
    """
    Tests our trained AI agent on the same map it was trained on
    Returns performance statistics
    """
    print("Loading model...")
    # Creating the brain with the same structure used in training
    ai_brain = DQN(in_states=4, h1_nodes=16, out_actions=4)
    ai_brain.load_state_dict(torch.load(dqn_path, weights_only=True))
    ai_brain.eval()

    all_results = []

    # Pre-calculate state tensors for all possible positions
    print("Pre-calculating state tensors...")
    state_tensors = {}
    for state in range(64):  # 8x8 grid
        row = state // 8
        col = state % 8
        goal_row = 7
        goal_col = 7
        row_diff = (goal_row - row)/7
        col_diff = (goal_col - col)/7
        state_tensors[state] = torch.tensor([[row/7, col/7, row_diff, col_diff]], dtype=torch.float32)

    for run in range(num_runs):
        print(f"\nStarting test run {run + 1}/{num_runs}...")
        
        # Create environment with the same 8x8 map used in training
        game_world = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=is_slippery)
        
        results = {
            'successes': [],
            'steps': [],
            'rewards': [],
            'success_steps': [],
            'fail_steps': [],
            'episode_rewards': []
        }

        for episode in range(episodes):
            current_position, _ = game_world.reset()
            game_over = False
            steps = 0
            total_reward = 0

            while not game_over:
                # Use pre-calculated state tensor
                state_tensor = state_tensors[current_position]

                # Ask AI for action (no exploration during testing)
                with torch.no_grad():
                    action = ai_brain(state_tensor).argmax().item()

                # Take action in the game world
                new_position, reward, terminated, truncated, _ = game_world.step(action)
                game_over = terminated or truncated
                
                # Update tracking
                current_position = new_position
                steps += 1
                total_reward += reward

            # Recording the results 
            won = 1 if reward == 1 else 0
            results['successes'].append(won)
            results['steps'].append(steps)
            results['rewards'].append(total_reward)
            results['episode_rewards'].append(total_reward)
            
            if won:
                results['success_steps'].append(steps)
            else:
                results['fail_steps'].append(steps)

            # Print progress every 100 episodes
            if (episode + 1) % 100 == 0:
                current_win_rate = np.mean(results['successes'][-100:]) * 100
                print(f"\nEpisodes {episode-99}-{episode+1}:")
                print(f"Win Rate: {current_win_rate:.1f}%")
                print(f"Average Steps: {np.mean(results['steps'][-100:]):.1f}")
        
        game_world.close()
        all_results.append(results)
        
        # Print results for this run
        stats = {
            'success_rate': np.mean(results['successes']),
            'avg_steps': np.mean(results['steps']),
            'avg_reward': np.mean(results['rewards']),
            'avg_success_steps': np.mean(results['success_steps']) if results['success_steps'] else 0,
            'avg_fail_steps': np.mean(results['fail_steps']) if results['fail_steps'] else 0,
            'episode_rewards': results['episode_rewards'],
            'steps': results['steps'],
            'successes': results['successes']
        }
        
        print(f"\nTest Run {run + 1} Results:")
        print(f"Win Rate: {stats['success_rate']*100:.1f}%")
        print(f"Average Steps: {stats['avg_steps']:.1f}")
        print(f"Successful Paths: {stats['avg_success_steps']:.1f} steps")
        print(f"Failed Paths: {stats['avg_fail_steps']:.1f} steps")

    # Calculate averages across all runs
    avg_stats = {
        'success_rate': np.mean([np.mean(r['successes']) for r in all_results]),
        'avg_steps': np.mean([np.mean(r['steps']) for r in all_results]),
        'avg_reward': np.mean([np.mean(r['rewards']) for r in all_results]),
        'avg_success_steps': np.mean([np.mean(r['success_steps']) if r['success_steps'] else 0 for r in all_results]),
        'avg_fail_steps': np.mean([np.mean(r['fail_steps']) if r['fail_steps'] else 0 for r in all_results]),
        'episode_rewards': [r['episode_rewards'] for r in all_results],
        'steps': [r['steps'] for r in all_results],
        'successes': [r['successes'] for r in all_results]
    }

    print("\nOverall Test Results (Averaged across all runs):")
    print(f"Win Rate: {avg_stats['success_rate']*100:.1f}%")
    print(f"Average Steps: {avg_stats['avg_steps']:.1f}")
    print(f"Successful Paths: {avg_stats['avg_success_steps']:.1f} steps")
    print(f"Failed Paths: {avg_stats['avg_fail_steps']:.1f} steps")
    
    # Plot the results
    plot_test_metrics(all_results)
    
    return avg_stats

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Run the evaluation
    print("Starting DQL evaluation...")
    stats = evaluate_frozen_lake(episodes=1000, num_runs=3)