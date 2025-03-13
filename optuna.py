import optuna
import numpy as np
from frozen_lake_dql import FrozenLakeDQL

def objective(trial: optuna.trial.Trial) -> float:
    """
    Samples hyperparameters, trains FrozenLakeDQL, 
    and returns a performance metric to be maximized.
    """
    #Ranges for our hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True)
    discount_factor = trial.suggest_float("discount_factor", 0.85, 0.99)
    network_sync_rate = trial.suggest_int("network_sync_rate", 5, 50)
    replay_memory_size = trial.suggest_categorical("replay_memory_size", [500, 1000, 3000, 5000])
    mini_batch_size = trial.suggest_categorical("mini_batch_size", [16, 32, 64])
    episodes = trial.suggest_int("episodes", 400, 1200, step=200)

    agent = FrozenLakeDQL(
        learning_rate_a=learning_rate,
        discount_factor_g=discount_factor,
        network_sync_rate=network_sync_rate,
        replay_memory_size=replay_memory_size,
        mini_batch_size=mini_batch_size
    )
    agent.train(episodes=episodes, render=False, is_slippery=False)

    ''' For measuring success, let's do the sum of rewards over the last 100 episodes
    # By default, we prefer a higher sum => "maximize"
    # If there is no recorded reward array, return 0
    # (in practice, agent.rewards_per_episode might not be stored as instance var,
    #  so let's retrieve it from local scope in train())
    # We'll do a quick fix: we can only retrieve rewards if we track them. 
    # Let's do the best we can:
    
    # We know from the code: the train() method used `rewards_per_episode` as local.
    # We can slightly modify that approach:
    # => We'll store them inside agent for final retrieval. 
    #    Or simply re-run the logic that was in train. 
    # For simplicity, let's assume we store it in agent.rewards_per_episode
    # (So let's patch the train() to store it in agent if needed)'''

    rewards_array = agent.rewards_per_episode
    
    if rewards_array is None or len(rewards_array) == 0:
        return 0.0
    
    last_N = min(100, len(rewards_array))
    score = np.sum(rewards_array[-last_N:])

    return score

if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10) 

    print("Best trial:")
    print(f"  Value (sum of rewards in last 100 episodes): {study.best_trial.value}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
