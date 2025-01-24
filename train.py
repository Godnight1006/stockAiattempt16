import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from trading_env import TradingEnv  # Import the trading environment

# Initialize the environment with action masking
env = TradingEnv(tickers=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'])  # Example stocks
env = ActionMasker(env, action_mask_fn=lambda env: env.get_action_masks())

from torch import nn
from stable_baselines3.common.callbacks import ProgressBarCallback
from matplotlib import pyplot as plt

model = MaskablePPO(
    "MultiInputPolicy",  # Changed from MlpPolicy
    env,
    learning_rate=3e-4,
    n_steps=256,  # Reduced from 2048
    batch_size=64,  # Keep same but now relative to n_steps
    n_epochs=1,  # Reduced from 10
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    policy_kwargs={
        "net_arch": [512, 256],  # Add this to match your transformer dimensions
        "activation_fn": nn.ReLU
    },
    vf_coef=0.5,
    max_grad_norm=0.5
)

# Start training with progress bar
model.learn(
    total_timesteps=5000,  # Reduced from 1,000,000
    callback=ProgressBarCallback()
)

# Save the trained model
model.save("ppo_trading_agent")

# Validation simulation --------------------------------------------------------
print("\nStarting validation...")
validation_env = TradingEnv(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'],
    initial_balance=100_000,  # New initial balance
    start_date='2024-01-01',  # Fixed dates
    end_date='2024-12-31'
)
validation_env = ActionMasker(validation_env, action_mask_fn=lambda env: env.get_action_masks())

obs, _ = validation_env.reset()
done = False
portfolio_values = []

while not done:
    action_masks = validation_env.get_action_masks()
    action, _ = model.predict(obs, action_masks=action_masks)
    obs, reward, terminated, truncated, info = validation_env.step(action)
    done = terminated or truncated
    portfolio_values.append(info['portfolio_value'])
    print(f"Step {len(portfolio_values)}: Portfolio Value ${info['portfolio_value']:.2f}")

if portfolio_values:
    print(f"\nFinal Portfolio Value: ${portfolio_values[-1]:.2f}")
    plt.figure(figsize=(10,6))
    plt.plot(portfolio_values)
    plt.title("2024 Validation Performance")
    plt.ylabel("Portfolio Value ($)")
    plt.xlabel("Trading Day")
    plt.show()
else:
    print("Validation failed - no steps completed")
