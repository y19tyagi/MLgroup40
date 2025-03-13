import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from frozen_lake_dql import DQN
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

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

    policy_dqn = DQN(in_states=states, h1_nodes=states, out_actions=actions)
    policy_dqn.load_state_dict(torch.load(dqn_path))
    policy_dqn.eval()

    for _ in range(episodes):
        random_map = generate_random_map(size=4, p=0.8)
        env = gym.make('FrozenLake-v1', desc=random_map, is_slippery=is_slippery, render_mode=None)
        state, _ = env.reset()
        terminated = False
        truncated = False
        episode_steps = 0
        episode_reward = 0.0

        while not (terminated or truncated):
            # One-hot representation of the state
            state_tensor = torch.zeros(states).unsqueeze(0)
            state_tensor[0, state] = 1.0

            with torch.no_grad():
                action = policy_dqn(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            episode_reward += reward
            episode_steps += 1

        # A reward of 1 means success (goal reached)
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

    # New metrics: average steps in successful episodes vs. failed episodes
    avg_steps_success = np.mean(steps_when_successful) if len(steps_when_successful) > 0 else float('nan')
    avg_steps_failed = np.mean(steps_when_failed) if len(steps_when_failed) > 0 else float('nan')

    print(f"Evaluation over {episodes} episodes:")
    print(f"  Success Rate: {overall_success_rate:.2f}")
    print(f"  Average Steps (all episodes): {overall_avg_steps:.2f}")
    print(f"  Average Reward (all episodes): {overall_avg_reward:.2f}")
    print(f"  Average Steps (only successful episodes): {avg_steps_success:.2f}")
    print(f"  Average Steps (only failed episodes): {avg_steps_failed:.2f}")

    # Return all stats for further use (e.g., plotting)
    return {
        "success_list": success_list,
        "steps_list": steps_list,
        "rewards_list": rewards_list,
        "overall_success_rate": overall_success_rate,
        "overall_avg_steps": overall_avg_steps,
        "overall_avg_reward": overall_avg_reward,
        "avg_steps_success": avg_steps_success,
        "avg_steps_failed": avg_steps_failed
    }

if __name__ == "__main__":
    metrics = evaluate_frozen_lake_extended(dqn_path="frozen_lake_dql.pt", episodes=100, is_slippery=False)

    # Example: plot success_list
    plt.plot(metrics["success_list"], 'o')
    plt.title("Success (1=goal) over episodes")
    plt.xlabel("Episode")
    plt.ylabel("Success")
    plt.show()
