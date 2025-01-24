import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.balance = 1000000  # Initial balance
        self.holdings = None
        self.current_prices = None
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = spaces.Dict({
            'balance': spaces.Box(low=0, high=np.inf, shape=(1,)),
            'holdings': spaces.Box(low=0, high=np.inf, shape=(1,)),
            'prices': spaces.Box(low=0, high=np.inf, shape=(1,))
        })
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        self.balance = 1000000
        self.holdings = [0] * len(self.current_prices) if self.current_prices else None
        # Return initial observation (placeholder)
        observation = {
            'balance': self.balance,
            'holdings': self.holdings,
            'prices': self.current_prices
        }
        return observation, {}  # Add empty info dict
    
    def step(self, actions):
        """Execute one time step in the environment"""
        if not isinstance(actions, (list, tuple)):
            actions = [actions]
            
        # Execute trades for each action
        for i, action in enumerate(actions):
            self.execute_trade(action, i)
            
        # Calculate portfolio value
        portfolio_value = self.balance + sum(
            h * p for h, p in zip(self.holdings, self.current_prices)
        )
        
        # Calculate reward (placeholder: portfolio return)
        reward = portfolio_value - 1000000  # Simple profit-based reward
        
        # Create observation
        observation = {
            'balance': self.balance,
            'holdings': self.holdings,
            'prices': self.current_prices
        }
        
        # Done is always False for continuous trading
        done = False
        
        # Info dictionary
        info = {
            'portfolio_value': portfolio_value,
            'positions': self.holdings,
            'action_masks': self.get_action_masks()  # Add action masks
        }
        
        return observation, reward, False, False, info  # (obs, reward, terminated, truncated, info)

    def action_mask(self, stock_idx: int) -> np.ndarray:
        """Generate action mask for a specific stock"""
        price = self.current_prices[stock_idx]
        can_buy = self.balance >= price and price > 0
        can_sell = self.holdings[stock_idx] > 0
        return np.array([can_buy, True, can_sell])  # [Buy, Hold, Sell]

    def get_action_masks(self) -> np.ndarray:
        """Get action masks for all stocks"""
        return np.array([self.action_mask(i) for i in range(len(self.current_prices))])

    def execute_trade(self, action, stock_idx):
        price = self.current_prices[stock_idx]
        max_risk = self.balance * 0.02
        max_shares = min(max_risk / price, self.balance * 0.1 / price)
        
        if action == 0:  # Buy
            shares = min(max_shares, self.balance / price)
            self.holdings[stock_idx] += shares
            self.balance -= shares * price
        elif action == 2:  # Sell 50%
            shares = self.holdings[stock_idx] * 0.5
            self.holdings[stock_idx] -= shares
            self.balance += shares * price
