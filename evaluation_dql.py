import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from frozen_lake_dql import DQN
from gym.envs.toy_text.frozen_lake import generate_random_map

def evaluate_frozen_lake(dqn_path="frozen_lake_dql.pt", episodes=100, is_slippery=False):
    """
    Tests our trained AI agent on new FrozenLake maps
    Returns performance statistics
    """
    # Creating the brain with the same structure used in training
    ai_brain = DQN(in_states=2, h1_nodes=16, out_actions=4)
    ai_brain.load_state_dict(torch.load(dqn_path))
    ai_brain.eval()  # Put brain in evaluation mode

    results = {
        'successes': [],
        'steps': [],
        'rewards': [],
        'success_steps': [],
        'fail_steps': []
    }

    for _ in range(episodes):
        # Create new random map
        game_world = gym.make('FrozenLake-v1', 
                             desc=generate_random_map(size=4),
                             is_slippery=is_slippery)
        current_position, _ = game_world.reset()
        game_over = False
        steps = 0
        total_reward = 0

        while not game_over:
           
            row = current_position // 4  # Map position to grid row
            col = current_position % 4   # Map position to grid column
            state_tensor = torch.tensor([[row/3, col/3]], dtype=torch.float32)

            # Ask AI for action
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
        
        if won:
            results['success_steps'].append(steps)
        else:
            results['fail_steps'].append(steps)
        
        game_world.close()

    stats = {
        'success_rate': np.mean(results['successes']),
        'avg_steps': np.mean(results['steps']),
        'avg_reward': np.mean(results['rewards']),
        'avg_success_steps': np.mean(results['success_steps']) if results['success_steps'] else 0,
        'avg_fail_steps': np.mean(results['fail_steps']) if results['fail_steps'] else 0
    }

    print("\nTest Results:")
    print(f"Success Rate: {stats['success_rate']*100:.1f}%")
    print(f"Average Steps: {stats['avg_steps']:.1f}")
    print(f"Successful Paths: {stats['avg_success_steps']:.1f} steps")
    print(f"Failed Paths: {stats['avg_fail_steps']:.1f} steps")
    
    return stats

if __name__ == "__main__":

    num_tests = 50
    test_history = {
        'success_rates': [],
        'all_steps': [],
        'success_steps': [],
        'fail_steps': []
    }

    print(f"Running {num_tests} tests...")
    for test_num in range(num_tests):
        print(f"Test {test_num+1}/{num_tests}")
        test_results = evaluate_frozen_lake(episodes=1000)
        
        test_history['success_rates'].append(test_results['success_rate'])
        test_history['all_steps'].append(test_results['avg_steps'])
        test_history['success_steps'].append(test_results['avg_success_steps'])
        test_history['fail_steps'].append(test_results['avg_fail_steps'])

    test_numbers = np.arange(1, num_tests+1)
    
    plt.figure(figsize=(10,6))
    plt.plot(test_numbers, test_history['success_rates'], 'b-')
    plt.title("Success Rate Progress")
    plt.xlabel("Test Number")
    plt.ylabel("Success Rate")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(test_numbers, test_history['all_steps'], 'g-')
    plt.title("Average Steps Per Test")
    plt.xlabel("Test Number")
    plt.ylabel("Steps")
    plt.grid(True)
    plt.show()