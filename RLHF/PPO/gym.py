import gymnasium as gym

# Initialise the environment
env = gym.make("CartPole-v1", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
print(info)
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    sample_action = 0 
    observation, reward, terminated, truncated, info = env.step(sample_action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:

        print('TERMINATED',terminated)
        #observation, info = env.reset()

env.close()
