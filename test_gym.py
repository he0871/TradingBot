from stable_baselines3 import PPO
from TradingEnv import TradingEnv
import matplotlib.pyplot as plt
import numpy as np



def test_model_on_env(model, env, env_name):
    """Test the model on a given environment and return results."""
    rewards_history = []
    cumulative_rewards = []
    steps = []
    
    obs = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    print(f"Testing on {env_name}...")
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        
        # Track the data for plotting
        rewards_history.append(reward)
        total_reward += reward
        cumulative_rewards.append(total_reward)
        steps.append(step_count)
        
        env.render()
        step_count += 1
    
    print(f"\n=== {env_name} Results ===")
    print(f"Final cumulative reward: {total_reward}")
    print(f"Total steps: {step_count}")
    print(f"Average reward per step: {total_reward/step_count if step_count > 0 else 0}")
    
    return rewards_history, cumulative_rewards, steps, total_reward


if __name__ == "__main__":
    # Initialize the trading environment
    training_env_path = "data/processed/"
    
    # Get all parquet files in the training_env_path
    
    training_env = TradingEnv(training_env_path)

    model = PPO("MlpPolicy", training_env, verbose=1)
    model.learn(total_timesteps=10000)

    test_env = TradingEnv("data/test_data/")
    
    # Test the model on both training and test datasets
    #train_rewards, train_cumulative, train_steps, train_total = test_model_on_env(model, env, "Training Dataset (2025-06-09)")
    test_rewards, test_cumulative, test_steps, test_total = test_model_on_env(model, test_env, "Test Dataset (2025-07-18)")

    # Save the model
    model.save("model/RL_trading_model")
    # Create comparison visualization
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 4)
    plt.plot(test_steps, test_cumulative, 'g-', linewidth=2)
    plt.title('Test Dataset Performance')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("trading_rewards_comparison.png", dpi=300, bbox_inches='tight')