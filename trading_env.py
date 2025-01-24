import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from feature_engineering import add_technical_indicators

class TradingEnv(gym.Env):
    def __init__(self, tickers=['AAPL', 'MSFT', 'GOOG']):
        super().__init__()
        self.tickers = tickers
        self.balance = 1_000_000
        self.holdings = None
        self.historical_data = None
        self.current_step = 0
        
        # Load 5 years of historical data
        self.start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = spaces.Dict({
            'features': spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.tickers), 30, 18)),  # Updated shape
            'balance': spaces.Box(low=0, high=np.inf, shape=(1,)),
            'holdings': spaces.Box(low=0, high=np.inf, shape=(len(self.tickers),))
        })
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        # Download and process data
        raw_data = yf.download(self.tickers, start=self.start_date, end=self.end_date, group_by='ticker')
        processed_data = [add_technical_indicators(raw_data[ticker]) for ticker in self.tickers]
        self.historical_data = processed_data
        self.current_step = 30  # Start after 30 days to have enough history
        self.balance = 1_000_000
        self.holdings = np.zeros(len(self.tickers))
        self.current_prices = np.array([data.iloc[self.current_step]['Close'] for data in self.historical_data])  # Initialize prices
        return self._get_observation(), {}

    def _get_observation(self):
        """Get 30-day window of features ending at current_step"""
        window_data = []
        for i in range(len(self.tickers)):
            features = self.historical_data[i].iloc[self.current_step-29:self.current_step+1][[
                'Norm_Close', 'Return_1', 'Return_5', 'Return_20',
                'RSI', 'MACD', 'Signal', 
                'Volatility_5', 'Volatility_20', 'Vol_Ratio',
                'Momentum_5', 'Momentum_20', 'Momentum_Ratio',
                'MA_20', 'MA_50', 'MA_200',
                'MA_20_50_Cross', 'MA_50_200_Cross'
            ]].values
            window_data.append(features)
        return {
            'features': np.array(window_data, dtype=np.float32),
            'balance': np.array([self.balance], dtype=np.float32),
            'holdings': self.holdings.copy().astype(np.float32)
        }
    
    def step(self, actions):
        """Execute one time step in the environment"""
        if not isinstance(actions, (list, tuple)):
            actions = [actions]
            
        # Get and store current prices
        current_prices = np.array([data.iloc[self.current_step]['Close'] for data in self.historical_data])
        # Check if we're at the end of the data
        done = self.current_step >= len(self.historical_data[0]) - 1
        if not done:
            self.current_prices = current_prices  # Keep existing assignment but wrap in condition
        else:
            self.current_prices = np.zeros(len(self.tickers))  # Add safety for terminal state
            
        # Execute trades for each action
        for i, action in enumerate(actions):
            self.execute_trade(action, i)
            
        # Calculate portfolio value
        portfolio_value = self.balance + sum(
            h * p for h, p in zip(self.holdings, current_prices)
        )
        
        # Calculate reward (placeholder: portfolio return)
        reward = portfolio_value - 1_000_000  # Simple profit-based reward
        
        # Advance to next time step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.historical_data[0]) - 1
        
        # Get new observation
        observation = self._get_observation()
        
        # Info dictionary
        info = {
            'portfolio_value': portfolio_value,
            'positions': self.holdings,
            'action_masks': self.get_action_masks()  # Add action masks
        }
        
        return observation, reward, done, done, info  # (obs, reward, terminated, truncated, info)

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
