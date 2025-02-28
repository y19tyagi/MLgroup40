import gym
import numpy as np

# Create the FrozenLake environment
env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human')

# Get the agent's current position from the state
def get_position(state, grid_size=8):
    row = state // grid_size  # integer division gives the row
    col = state % grid_size   # modulus gives the column
    return row, col

# Run the environment
state = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # Sample a random action
    next_state, reward, done, truncated, info = env.step(action)
    
    # Get the position of the agent from the current state
    position = get_position(next_state)  # Extract position from the state
    print(f"Agent position: {position}")
    
    env.render()  # Render the environment (visualization)
   
env.close()
