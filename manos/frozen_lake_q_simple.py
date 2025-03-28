import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pickle  # Add this import at the top
from datetime import datetime

class FrozenLakeAgent:
    def __init__(self):
        # Create environment - making it non-slippery for better learning
        self.env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
        
        # Initialize Q-table with slightly optimistic values
        self.q_table = np.ones((64, 4)) * 0.1  # Less optimistic initialization
        
        # Learning parameters
        self.learning_rate = 0.2    # Lower learning rate for more stable learning
        self.discount = 0.99        # Keep high discount
        self.initial_explore = 1.0  # Start with full exploration
        self.min_explore = 0.01     # Keep low minimum
        self.explore_chance = self.initial_explore
        self.explore_decay = 0.995  # Slower exploration decay
        
        # Training tracking
        self.games_played = 0
        self.wins = 0
        self.window_size = 50
        
        # History for plotting
        self.win_rates = []
        self.rewards = []
        self.steps_history = []
        self.explore_rates = []
        
        # Moving averages
        self.reward_window = deque(maxlen=self.window_size)
        self.steps_window = deque(maxlen=self.window_size)
    
    def get_manhattan_distance(self, state):
        current_row = state // 8
        current_col = state % 8
        goal_row = 7
        goal_col = 7
        return abs(current_row - goal_row) + abs(current_col - goal_col)
    
    def calculate_reward(self, reward, state):
        # Base reward from environment
        if reward == 1:
            return 1000.0  # Even bigger reward for reaching goal
        elif reward == 0:
            current_distance = self.get_manhattan_distance(state)
            max_distance = 14  # Maximum possible Manhattan distance in 8x8 grid
            
            # Enhanced distance-based reward shaping
            distance_reward = 5.0 * (max_distance - current_distance) / max_distance
            
            # Progressive penalties based on distance
            if current_distance > 10:
                return -0.5 + distance_reward
            elif current_distance > 5:
                return -0.2 + distance_reward
            
            return distance_reward
        else:
            return -10.0  # Bigger penalty for falling in hole
    
    def play_game(self):
        state = self.env.reset()[0]
        total_reward = 0
        steps = 0
        
        for _ in range(200):  # Increased max steps per game
            # Choose action with epsilon-greedy strategy
            if np.random.random() < self.explore_chance:
                action = self.env.action_space.sample()
            else:
                # Add small random noise to break ties
                noise = np.random.random((4,)) * 0.01
                action = np.argmax(self.q_table[state] + noise)
            
            # Take action
            new_state, reward, done, _, _ = self.env.step(action)
            
            # Calculate shaped reward
            shaped_reward = self.calculate_reward(reward, state)
            total_reward += shaped_reward
            
            # Update Q-table
            old_value = self.q_table[state, action]
            next_max = np.max(self.q_table[new_state])
            new_value = old_value + self.learning_rate * (
                shaped_reward + self.discount * next_max - old_value
            )
            self.q_table[state, action] = new_value
            
            steps += 1
            state = new_state
            
            if done:
                if reward == 1:
                    self.wins += 1
                self.games_played += 1
                break
        
        # Increment games_played if we hit step limit without finishing
        if steps == 200:
            self.games_played += 1
        
        # Update moving averages
        self.reward_window.append(total_reward)
        self.steps_window.append(steps)
        
        # Decay exploration rate
        self.explore_chance = max(self.min_explore, 
                                self.explore_chance * self.explore_decay)
        
        return total_reward, steps
    
    def plot_progress(self):
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Win Rate
        plt.subplot(2, 2, 1)
        plt.plot(self.win_rates)
        plt.title('Win Rate Over Time')
        plt.xlabel('Games')
        plt.ylabel('Win Rate (%)')
        
        # Plot 2: Rewards
        plt.subplot(2, 2, 2)
        plt.plot(self.rewards)
        plt.title('Rewards Over Time')
        plt.xlabel('Games')
        plt.ylabel('Total Reward')
        
        # Plot 3: Steps
        plt.subplot(2, 2, 3)
        plt.plot(self.steps_history)
        plt.title('Steps Per Game')
        plt.xlabel('Games')
        plt.ylabel('Steps')
        
        # Plot 4: Exploration Rate
        plt.subplot(2, 2, 4)
        plt.plot(self.explore_rates)
        plt.title('Exploration Rate')
        plt.xlabel('Games')
        plt.ylabel('Explore Rate')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()
    
    def train(self, num_games=1000):
        print("Starting training...")
        
        for game in range(num_games):
            reward, steps = self.play_game()
            
            # Update progress tracking
            win_rate = (self.wins / self.games_played) * 100
            self.win_rates.append(win_rate)
            self.rewards.append(reward)
            self.steps_history.append(steps)
            self.explore_rates.append(self.explore_chance)
            
            # Show progress every 100 games
            if (game + 1) % 100 == 0:
                avg_reward = np.mean(list(self.reward_window))
                avg_steps = np.mean(list(self.steps_window))
                
                print(f"\nGames played: {self.games_played}")
                print(f"Win rate: {win_rate:.1f}%")
                print(f"Average reward: {avg_reward:.3f}")
                print(f"Average steps: {avg_steps:.1f}")
                print(f"Explore rate: {self.explore_chance:.3f}")
        
        print("\nTraining completed. Generating plot...")
        self.plot_progress()
        print("Plot saved as 'training_progress.png'")

    def save_model(self, filename='frozen_lake_model.pkl'):
        """Save the trained model to a file"""
        model_data = {
            'q_table': self.q_table,
            'games_played': self.games_played,
            'wins': self.wins,
            'explore_chance': self.explore_chance
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename='frozen_lake_model.pkl'):
        """Load a trained model from a file"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = model_data['q_table']
            self.games_played = model_data['games_played']
            self.wins = model_data['wins']
            self.explore_chance = model_data['explore_chance']
            print(f"Model loaded from {filename}")
        except FileNotFoundError:
            print(f"No model file found at {filename}")
            return False
        return True

    def plot_test_results(self, all_results):
        """Plot the test results from all runs"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Win Rate Over Time (for each run)
        plt.subplot(2, 2, 1)
        colors = ['b', 'r', 'g']
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
        
        # Save the plot with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'results/q_learning_test_results_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nTest results plot saved as '{filename}'")

    def test(self, num_episodes=1000, num_runs=3):
        """
        Test the agent on the same map it was trained on, multiple times
        Returns statistics about the test runs
        """
        all_results = []
        
        for run in range(num_runs):
            print(f"\nStarting test run {run + 1}/{num_runs}...")
            
            # Create test environment with same settings as training
            test_env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
            
            results = {
                'successes': [],
                'steps': [],
                'rewards': [],
                'success_steps': [],
                'fail_steps': [],
                'episode_rewards': []
            }
            
            for episode in range(num_episodes):
                state = test_env.reset()[0]
                done = False
                steps = 0
                total_reward = 0
                
                while not done:
                    # Use greedy policy (no exploration during testing)
                    action = np.argmax(self.q_table[state])
                    state, reward, done, _, _ = test_env.step(action)
                    steps += 1
                    total_reward += reward
                    
                    if done:
                        if reward == 1:
                            results['success_steps'].append(steps)
                        else:
                            results['fail_steps'].append(steps)
                        break
                
                # Record results
                won = 1 if reward == 1 else 0
                results['successes'].append(won)
                results['steps'].append(steps)
                results['rewards'].append(total_reward)
                results['episode_rewards'].append(total_reward)
                
                # Print progress every 100 episodes
                if (episode + 1) % 100 == 0:
                    current_win_rate = np.mean(results['successes'][-100:]) * 100
                    print(f"\nEpisodes {episode-99}-{episode+1}:")
                    print(f"Win Rate: {current_win_rate:.1f}%")
                    print(f"Average Steps: {np.mean(results['steps'][-100:]):.1f}")
            
            test_env.close()
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
        self.plot_test_results(all_results)
        
        return avg_stats

if __name__ == "__main__":
    # Create agent and load the trained model
    agent = FrozenLakeAgent()
    agent.load_model('frozen_lake_model.pkl')
    
    # Test the model
    print("\nTesting the model...")
    test_stats = agent.test(num_episodes=1000, num_runs=3) 