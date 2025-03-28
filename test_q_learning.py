import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from manos import frozen_lake_q_simple
import os
from datetime import datetime

def ensure_results_dir():
    """Create a results directory if it doesn't exist"""
    if not os.path.exists('results'):
        os.makedirs('results')

def plot_test_metrics(stats_list):
    """Plot test metrics for all three runs"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = ['b', 'r', 'g']
    
    for i, stats in enumerate(stats_list):
        # Plot 1: Success Rate
        success_rate = np.cumsum(stats['successes']) / np.arange(1, len(stats['successes']) + 1) * 100
        ax1.plot(success_rate, f'{colors[i]}-', label=f'Test {i+1}')
        ax1.set_title('Success Rate Over Time')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Success Rate (%)')
        ax1.grid(True)
        ax1.set_ylim(0, 100)
        ax1.legend()
        
        # Plot 2: Episode Rewards
        ax2.plot(stats['episode_rewards'], f'{colors[i]}-', label=f'Test {i+1}')
        ax2.set_title('Episode Rewards')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax2.grid(True)
        ax2.legend()
        
        # Plot 3: Steps per Episode
        ax3.plot(stats['steps'], f'{colors[i]}-', label=f'Test {i+1}')
        ax3.set_title('Steps per Episode')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Steps')
        ax3.grid(True)
        ax3.legend()
        
        # Plot 4: Moving Average Success Rate
        window = 100
        success_rates = np.array(stats['successes']) * 100
        moving_avg = np.convolve(success_rates, np.ones(window)/window, mode='valid')
        ax4.plot(moving_avg, f'{colors[i]}-', label=f'Test {i+1}')
        ax4.set_title(f'Moving Average Success Rate (window={window})')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Success Rate (%)')
        ax4.grid(True)
        ax4.set_ylim(0, 100)
        ax4.legend()
    
    plt.tight_layout()
    
    # Save the plot with timestamp and success rate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    success_rate = np.mean([stats['success_rate'] for stats in stats_list]) * 100
    filename = f'results/q_learning_test_{success_rate:.1f}percent_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return filename

if __name__ == "__main__":
    # Create results directory
    ensure_results_dir()
    
    # Load the trained agent
    print("Loading trained model...")
    agent = frozen_lake_q_simple.FrozenLakeAgent()
    agent.load_model('frozen_lake_model.pkl')
    
    # Run three tests
    stats_list = []
    for test_num in range(3):
        print(f"\nRunning Test {test_num + 1}/3...")
        stats = agent.test(num_episodes=1000)
        stats_list.append(stats)
    
    # Plot results
    print("\nGenerating plots...")
    saved_file = plot_test_metrics(stats_list)
    print(f"Plots saved as '{saved_file}'") 