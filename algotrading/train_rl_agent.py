import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from TradingEnv import TradingEnv

# Create log directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

# Create and wrap the environment
def make_env():
    return TradingEnv(processed_data_path="data/processed")

env = DummyVecEnv([make_env])

# Initialize the agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
)

# Train the agent
total_timesteps = 100000  # Adjust based on your needs
print(f"Training for {total_timesteps} timesteps...")
model.learn(total_timesteps=total_timesteps, progress_bar=True)

# Save the trained model
model_path = os.path.join(model_dir, "ppo_trading_model")
model.save(model_path)
print(f"Model saved to {model_path}")

# Create a new environment for evaluation
eval_env = make_env()

# Evaluate the trained policy
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Test the model with a few episodes
obs, _ = eval_env.reset()
done = False
truncated = False
total_reward = 0
step = 0

print("\nTesting the trained model:")
eval_env.render()

while not done and not truncated and step < 100:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = eval_env.step(action)
    total_reward += reward
    
    if step % 10 == 0 or done:
        print(f"Step {step}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
        eval_env.render()
    
    step += 1

print(f"\nTest episode finished with total reward: {total_reward:.2f}")