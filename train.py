import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from trading_env import TradingEnv  # Import the trading environment

# Initialize the environment with action masking
env = TradingEnv(
    tickers=['AAPL', 'MSFT', 'AMZN', 'IBM', 'GE'],  # Companies with data since 2000
    start_date='2000-01-01',  # Changed from 2018 to get 24 years of data
    end_date='2023-12-31'
)
env = ActionMasker(env, action_mask_fn=lambda env: env.unwrapped.get_action_masks())

# Create validation env for early stopping
val_env = TradingEnv(
    tickers=['AAPL', 'MSFT', 'AMZN', 'IBM', 'GE'],  # Companies with data since 2000
    initial_balance=100_000,
    start_date='2024-01-01',
    end_date='2024-12-31'
)
val_env = ActionMasker(val_env, action_mask_fn=lambda env: env.unwrapped.get_action_masks())

from torch import nn
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
from matplotlib import pyplot as plt
import numpy as np

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, eval_env, patience=5, eval_freq=50_000):
        super().__init__()
        self.eval_env = eval_env
        self.patience = patience
        self.eval_freq = eval_freq
        self.best_value = -np.inf
        self.wait_count = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            current_value = self.evaluate_model()
            if current_value > self.best_value:
                self.best_value = current_value
                self.wait_count = 0
            else:
                self.wait_count += 1
                if self.wait_count >= self.patience:
                    print(f"\nEarly stopping at {self.n_calls} timesteps")
                    return False
        return True

    def evaluate_model(self):
        obs, _ = self.eval_env.reset()
        done = False
        final_value = None
        while not done:
            action_masks = self.eval_env.unwrapped.get_action_masks()
            action, _ = self.model.predict(obs, action_masks=action_masks)
            obs, _, done, _, info = self.eval_env.step(action)
            final_value = info['portfolio_value']
        return final_value

model = MaskablePPO(
    "MultiInputPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,  # Restore original
    batch_size=64,
    n_epochs=10,  # Restore original
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    policy_kwargs={
        "net_arch": [512, 256],
        "activation_fn": nn.ReLU
    },
    vf_coef=0.5,
    max_grad_norm=0.5
)

# Start training with progress bar and early stopping
model.learn(
    total_timesteps=1_000_000,  # Restore original
    callback=[ProgressBarCallback(), EarlyStoppingCallback(val_env)],
    reset_num_timesteps=False
)

# Save the trained model
model.save("ppo_trading_agent")

# Validation simulation --------------------------------------------------------
print("\nStarting validation...")
validation_env = TradingEnv(
    tickers=['AAPL', 'MSFT', 'AMZN', 'IBM', 'GE'],  # Companies with data since 2000
    initial_balance=100_000,  # New initial balance
    start_date='2024-01-01',  # Fixed dates
    end_date='2024-12-31'
)
validation_env = ActionMasker(validation_env, action_mask_fn=lambda env: env.unwrapped.get_action_masks())

obs, _ = validation_env.reset()
done = False
portfolio_values = []

while not done:
    action_masks = validation_env.unwrapped.get_action_masks()
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
