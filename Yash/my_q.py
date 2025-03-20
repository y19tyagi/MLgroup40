import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

class Qlearning_agent:
    def __init__(self, render_mode=False):
        # Create the FrozenLake environment with an 8x8 grid.
        self.env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True,
                            render_mode='human' if render_mode else None)
        self.num_episodes = 15000
        self.qtable = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.learn_rate = 0.1
        self.discount = 0.9
        self.epsilon = 1.0
        self.epsilon_decay_rate = 0.001
        self.random_num = np.random.default_rng()
        self.rewards_episodes = np.zeros(self.num_episodes)

    def learning(self):
        count = 0
        print("Start Training:")
        for episode in range(self.num_episodes):
            count += 1
            print(f"Episode: {count}")
            # Reset the environment and get the initial state.
            state, _ = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Choose action using epsilon-greedy strategy.
                if self.random_num.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.qtable[state,:])
                
                new_state, reward, done, truncated, _ = self.env.step(action)
                # Consider the episode done if either done or truncated is True.
                done = done or truncated

                # Q-learning update rule.
                self.qtable[state, action] += self.learn_rate * (
                    reward + self.discount * np.max(self.qtable[new_state]) - self.qtable[state, action]
                )
                state = new_state
                episode_reward += reward

            self.rewards_episodes[episode] = episode_reward

            # Decay epsilon after each episode.
            self.epsilon = max(self.epsilon - self.epsilon_decay_rate, 0)
            # Optionally adjust the learning rate when exploration is complete.
            if self.epsilon == 0:
                self.learn_rate = 0.0001

        # After training, plot the moving average of rewards.
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

        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()
        print("\nTesting the agent...")
        wins = 0
        test_rewards = []
        for test in range(num_tests):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                # Select the best action from the learned Q-table.
                action = np.argmax(q[state])
                state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated
                episode_reward += reward
            test_rewards.append(episode_reward)
            # Count a win if the reward in the final step was 1.
            wins += reward

        # Plot test rewards over episodes.
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
    agent.save_model()
    agent.test()
    
