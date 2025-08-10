import numpy as np
from TradingEnv import TradingEnv

# Create the trading environment
env = TradingEnv(processed_data_path="data/processed")

# Test reset
obs, _ = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"First observation: {obs}")

# Test a few steps with random actions
num_episodes = 2
max_steps = 100

for episode in range(num_episodes):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    truncated = False
    step = 0
    
    print(f"\nEpisode {episode+1}")
    env.render()
    
    while not done and not truncated and step < max_steps:
        # Take random action
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Print info every 10 steps
        if step % 10 == 0 or done:
            print(f"Step {step}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
            env.render()
        
        step += 1
    
    print(f"Episode {episode+1} finished after {step} steps with total reward: {total_reward:.2f}")

print("\nTesting complete!")