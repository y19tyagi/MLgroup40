import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from frozen_lake_dql import DQN
from gym.envs.toy_text.frozen_lake import generate_random_map

def evaluate_frozen_lake_extended(
    dqn_path="frozen_lake_dql.pt",
    episodes=100,
    is_slippery=False
):
    """
    Loads a trained DQN model and evaluates its performance on FrozenLake over multiple random maps.
    Returns a dictionary with per-episode metrics and overall averages.
    Additionally calculates average steps only among successful episodes.
    """
    success_list = []
    steps_list = []
    rewards_list = []
    steps_when_successful = []
    steps_when_failed = []

    states = 16
    actions = 4

    # Load the trained DQN model
    policy_dqn = DQN(in_states=states, h1_nodes=states, out_actions=actions)
    policy_dqn.load_state_dict(torch.load(dqn_path))
    policy_dqn.eval()

    for _ in range(episodes):
        env = gym.make('FrozenLake-v1', desc=generate_random_map(size=4),
                       is_slippery=is_slippery, render_mode=None)
        state, _ = env.reset()
        terminated = False
        truncated = False
        episode_steps = 0
        episode_reward = 0.0

        while not (terminated or truncated):
            # One-hot encode the state
            state_tensor = torch.zeros(states).unsqueeze(0)
            state_tensor[0, state] = 1.0

            with torch.no_grad():
                action = policy_dqn(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            episode_reward += reward
            episode_steps += 1

        # Determine success based on reward (1 means goal reached)
        success = 1 if reward == 1.0 else 0
        success_list.append(success)
        steps_list.append(episode_steps)
        rewards_list.append(episode_reward)

        if success == 1:
            steps_when_successful.append(episode_steps)
        else:
            steps_when_failed.append(episode_steps)

        env.close()

    overall_success_rate = np.mean(success_list)
    overall_avg_steps = np.mean(steps_list)
    overall_avg_reward = np.mean(rewards_list)
    avg_steps_success = np.mean(steps_when_successful) if steps_when_successful else float('nan')
    avg_steps_failed = np.mean(steps_when_failed) if steps_when_failed else float('nan')

    print(f"Evaluation over {episodes} episodes:")
    print(f"  Success Rate: {overall_success_rate:.2f}")
    print(f"  Average Steps (all episodes): {overall_avg_steps:.2f}")
    print(f"  Average Reward (all episodes): {overall_avg_reward:.2f}")
    print(f"  Average Steps (only successful episodes): {avg_steps_success:.2f}")
    print(f"  Average Steps (only failed episodes): {avg_steps_failed:.2f}")

    return {
        "overall_success_rate": overall_success_rate,
        "overall_avg_steps": overall_avg_steps,
        "overall_avg_reward": overall_avg_reward,
        "avg_steps_success": avg_steps_success,
        "avg_steps_failed": avg_steps_failed
    }

if __name__ == "__main__":
    num_evaluations = 50

    # Lists to store metrics for each evaluation
    evaluation_success_rates = []
    evaluation_avg_steps = []
    evaluation_avg_rewards = []
    evaluation_avg_steps_success = []
    evaluation_avg_steps_failed = []

    # Loop over 50 evaluations
    for i in range(num_evaluations):
        print(f"Running evaluation {i+1}/{num_evaluations}")
        metrics = evaluate_frozen_lake_extended(dqn_path="frozen_lake_dql.pt",
                                                  episodes=1000,
                                                  is_slippery=False)
        evaluation_success_rates.append(metrics["overall_success_rate"])
        evaluation_avg_steps.append(metrics["overall_avg_steps"])
        evaluation_avg_rewards.append(metrics["overall_avg_reward"])
        evaluation_avg_steps_success.append(metrics["avg_steps_success"])
        evaluation_avg_steps_failed.append(metrics["avg_steps_failed"])

    evaluations = np.arange(1, num_evaluations + 1)

    # Plot Success Rate over evaluations
    plt.figure()
    plt.plot(evaluations, evaluation_success_rates, marker='o', linestyle='-')
    plt.title("Success Rate Over Evaluations")
    plt.xlabel("Evaluation")
    plt.ylabel("Success Rate")
    plt.grid(True)
    plt.show()

    # Plot Average Steps (all episodes) over evaluations
    plt.figure()
    plt.plot(evaluations, evaluation_avg_steps, marker='o', linestyle='-')
    plt.title("Average Steps (All Episodes) Over Evaluations")
    plt.xlabel("Evaluation")
    plt.ylabel("Average Steps")
    plt.grid(True)
    plt.show()

    # Plot Average Reward (all episodes) over evaluations
    plt.figure()
    plt.plot(evaluations, evaluation_avg_rewards, marker='o', linestyle='-')
    plt.title("Average Reward (All Episodes) Over Evaluations")
    plt.xlabel("Evaluation")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.show()

    # Plot Average Steps (successful episodes) over evaluations
    plt.figure()
    plt.plot(evaluations, evaluation_avg_steps_success, marker='o', linestyle='-')
    plt.title("Average Steps (Successful Episodes) Over Evaluations")
    plt.xlabel("Evaluation")
    plt.ylabel("Average Steps (Success)")
    plt.grid(True)
    plt.show()

    # Plot Average Steps (failed episodes) over evaluations
    plt.figure()
    plt.plot(evaluations, evaluation_avg_steps_failed, marker='o', linestyle='-')
    plt.title("Average Steps (Failed Episodes) Over Evaluations")
    plt.xlabel("Evaluation")
    plt.ylabel("Average Steps (Failed)")
    plt.grid(True)
    plt.show()
