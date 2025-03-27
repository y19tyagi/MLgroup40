import optuna
import numpy as np
from frozen_lake_dql import FrozenLakeDQL

def objective(trial: optuna.trial.Trial) -> float:
    """
    Optimized hyperparameter sampling for FrozenLake DQN training
    Returns average reward from last 100 episodes (smoothed performance)
    """

    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    discount_factor = trial.suggest_float("discount_factor", 0.90, 0.99)
    network_sync_rate = trial.suggest_int("network_sync_rate", 50, 1000, step=50)
    replay_memory_size = trial.suggest_categorical("replay_memory_size", [5000, 10000, 20000])
    mini_batch_size = trial.suggest_categorical("mini_batch_size", [64, 128, 256])
    episodes = trial.suggest_int("episodes", 800, 2000, step=400)

    agent = FrozenLakeDQL(
        learning_rate_a=learning_rate,
        discount_factor_g=discount_factor,
        network_sync_rate=network_sync_rate,
        replay_memory_size=replay_memory_size,
        mini_batch_size=mini_batch_size,
    )

    agent.train(episodes=episodes, render=False, is_slippery=False)
    
    # These lines are calculating the smoothed performance metric
    rewards_array = agent.rewards_per_episode
    if not rewards_array.size:
        return 0.0
    
    # Use average of last 100 episodes instead of sum
    last_N = min(100, len(rewards_array))
    average_score = np.mean(rewards_array[-last_N:])
    
    # Optional: Add pruning for bad trials
    trial.report(average_score, step=episodes)
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    return average_score

if __name__ == "__main__":

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )
    
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print("\nBest trial:")
    print(f"  Average reward (last 100 episodes): {study.best_trial.value:.2f}")
    print("  Optimal parameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")