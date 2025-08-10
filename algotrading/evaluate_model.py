import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from TradingEnv import TradingEnv

# Load the trained model
model_path = os.path.join("model", "ppo_trading_model")
model = PPO.load(model_path)

# Create environment
env = TradingEnv(processed_data_path="data/processed")

# Run multiple evaluation episodes
num_episodes = 5
all_rewards = []
all_cash_histories = []
all_position_histories = []
all_price_histories = []
all_dates = []

for episode in range(num_episodes):
    # Reset environment
    obs, _ = env.reset()
    done = False
    truncated = False
    episode_reward = 0
    
    # Track history
    cash_history = [10000]  # Starting cash
    position_history = [0]  # Starting with no position
    price_history = []
    date_history = []
    
    step = 0
    
    print(f"\nEpisode {episode+1}")
    
    while not done and not truncated and step < 100:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        
        # Record history
        cash_history.append(info['cash'])
        position_history.append(info['position'])
        price_history.append(info['current_price'])
        
        # Get date if available
        if step < len(env.df):
            date_history.append(env.df.iloc[step]['timestamp'])
        
        # Print progress
        if step % 20 == 0 or done:
            print(f"Step {step}, Action: {action}, Reward: {reward:.2f}, Total Reward: {episode_reward:.2f}")
        
        step += 1
    
    print(f"Episode {episode+1} finished with total reward: {episode_reward:.2f}")
    
    # Store episode data
    all_rewards.append(episode_reward)
    all_cash_histories.append(cash_history)
    all_position_histories.append(position_history)
    all_price_histories.append(price_history)
    all_dates.append(date_history)

# Print overall statistics
print(f"\nAverage reward over {num_episodes} episodes: {np.mean(all_rewards):.2f}")
print(f"Best episode reward: {np.max(all_rewards):.2f}")
print(f"Worst episode reward: {np.min(all_rewards):.2f}")

# Plot results
plt.figure(figsize=(15, 10))

# Plot 1: Cash over time for each episode
plt.subplot(2, 2, 1)
for i, cash_history in enumerate(all_cash_histories):
    plt.plot(cash_history, label=f'Episode {i+1}')
plt.title('Cash Over Time')
plt.xlabel('Steps')
plt.ylabel('Cash')
plt.legend()

# Plot 2: Price and Position for best episode
best_episode = np.argmax(all_rewards)
plt.subplot(2, 2, 2)

# Create a twin axis for position
ax1 = plt.gca()
ax2 = ax1.twinx()

# Plot price
ax1.plot(all_price_histories[best_episode], 'b-', label='Price')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Price', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Plot position
ax2.plot(all_position_histories[best_episode], 'r-', label='Position')
ax2.set_ylabel('Position', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_ylim(-0.1, 1.1)  # Position is either 0 or 1

plt.title(f'Price and Position for Best Episode (Reward: {all_rewards[best_episode]:.2f})')

# Plot 3: Rewards per episode
plt.subplot(2, 2, 3)
plt.bar(range(1, num_episodes + 1), all_rewards)
plt.title('Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

# Plot 4: Cash change for best episode
plt.subplot(2, 2, 4)
plt.plot(all_cash_histories[best_episode])
plt.title(f'Cash for Best Episode (Reward: {all_rewards[best_episode]:.2f})')
plt.xlabel('Steps')
plt.ylabel('Cash')

plt.tight_layout()
plt.savefig('evaluation_results.png')
#plt.show()

# Save detailed results to CSV
results = {
    'Episode': [],
    'Total Reward': [],
    'Final Cash': [],
    'Trades Made': []
}

for i in range(num_episodes):
    # Count position changes as trades
    position_changes = sum(1 for j in range(1, len(all_position_histories[i])) 
                         if all_position_histories[i][j] != all_position_histories[i][j-1])
    
    results['Episode'].append(i+1)
    results['Total Reward'].append(all_rewards[i])
    results['Final Cash'].append(all_cash_histories[i][-1])
    results['Trades Made'].append(position_changes)

results_df = pd.DataFrame(results)
results_df.to_csv('evaluation_results.csv', index=False)
print(f"\nDetailed results saved to evaluation_results.csv")
print(f"Plots saved to evaluation_results.png")