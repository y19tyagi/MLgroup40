import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

class Qlearning_agent:
    def __init__(self, render_mode=False):
        # For debugging, we use a non-slippery (deterministic) version.
        self.env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False,
                            render_mode='human' if render_mode else None)
        self.num_episodes = 3000  # Increase episodes for better exploration.
        self.qtable = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.learn_rate = 0.1
        self.discount = 0.9
        self.epsilon = 1.0
        self.epsilon_decay_rate = 0.001
        self.random_num = np.random.default_rng()
        self.rewards_episodes = np.zeros(self.num_episodes)

    def learning(self):
        print("Start Training:")
        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Epsilon-greedy action selection.
                if self.random_num.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.qtable[state])
                
                new_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated

                # Q-learning update.
                self.qtable[state, action] += self.learn_rate * (
                    reward + self.discount * np.max(self.qtable[new_state]) - self.qtable[state, action]
                )
                state = new_state
                episode_reward += reward

            self.rewards_episodes[episode] = episode_reward

            # Optionally print when a win is obtained.
            if episode_reward > 0:
                print(f"Episode {episode+1}: Reached goal! Reward: {episode_reward}")

            # Decay epsilon after each episode.
            self.epsilon = max(self.epsilon - self.epsilon_decay_rate, 0)
            # Optionally adjust learning rate when exploration is finished.
            if self.epsilon == 0:
                self.learn_rate = 0.0001

        # Plot moving average of rewards.
        window = 100
        moving_avg = np.convolve(self.rewards_episodes, np.ones(window) / window, mode='valid')
        plt.figure()
        plt.plot(moving_avg)
        plt.xlabel("Episode")
        plt.ylabel("Moving Average Reward (window = 100)")
        plt.title("Training Performance on FrozenLake")
        plt.savefig('frozen_lake8x8.png')
        plt.show()

    def test(self, num_tests=100):
        print("\nTesting the agent...")
        wins = 0
        test_rewards = []
        for test in range(num_tests):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = np.argmax(self.qtable[state])
                state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated
                episode_reward += reward
            test_rewards.append(episode_reward)
            wins += 1 if episode_reward > 0 else 0

        plt.figure()
        plt.plot(test_rewards)
        plt.xlabel("Test Episode")
        plt.ylabel("Reward")
        plt.title("Test Performance on FrozenLake")
        plt.savefig("frozen_lake8x8_test_result.png")
        plt.show()

        success_rate = (wins / num_tests) * 100
        print(f"Success rate: {success_rate}%")

    def save_model(self):
        with open("frozen_lake8x8.pkl", "wb") as file:
            pickle.dump(self.qtable, file)

if __name__ == "__main__":
    agent = Qlearning_agent(render_mode=False)
    agent.learning()
    agent.test()
    agent.save_model()
