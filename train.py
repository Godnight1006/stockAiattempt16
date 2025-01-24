from sb3_contrib import MaskablePPO
from trading_env import TradingEnv  # Import the trading environment

# Initialize the environment
env = TradingEnv() 

from torch import nn

model = MaskablePPO(
    "MultiInputPolicy",  # Changed from MlpPolicy
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    policy_kwargs={
        "net_arch": [512, 256],  # Add this to match your transformer dimensions
        "activation_fn": nn.ReLU
    }
)
