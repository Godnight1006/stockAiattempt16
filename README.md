# Stock Trading AI

An advanced stock trading AI system that uses reinforcement learning to make trading decisions across multiple stocks. The system employs a Proximal Policy Optimization (PPO) algorithm with action masking and includes comprehensive technical analysis features.

## Features

- **Multi-Stock Trading**: Simultaneously trade multiple stocks (default: AAPL, MSFT, AMZN, IBM, GE)
- **Advanced Feature Engineering**:
  - Technical indicators (RSI, MACD, Moving Averages)
  - Volatility metrics
  - Momentum indicators
  - Price normalization
  - Moving average crossovers
- **Reinforcement Learning Model**:
  - Maskable PPO implementation
  - Decision Transformer architecture
  - Early stopping capability
  - Customizable neural network architecture
- **Risk Management**:
  - Position size limits
  - Portfolio balance constraints
  - Action masking for valid trades
- **Comprehensive Environment**:
  - Built on OpenAI Gymnasium
  - Real market data via yfinance
  - Flexible date ranges for training/testing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Godnight1006/stockAiattempt16.git
cd stockAiattempt16
```

2. Install required dependencies:
```bash
pip install torch numpy pandas yfinance gymnasium sb3_contrib matplotlib
```

## Project Structure

- `trading_env.py`: Custom OpenAI Gymnasium environment for stock trading
- `feature_engineering.py`: Technical indicator calculations and data preprocessing
- `model.py`: Neural network architecture definition
- `train.py`: Training and validation pipeline
- `simulator.py`: Trading simulation functionality
- `visualize.py`: Trading visualization utilities

## Usage

### Training

To train a new model:
```bash
python train.py
```

To skip training and run validation only:
```bash
python train.py --no-training
```

### Configuration

The system can be configured through various parameters:

1. **Trading Environment** (`trading_env.py`):
   - Modify `tickers` list for different stocks
   - Adjust `initial_balance` for different starting capital
   - Change date ranges using `start_date` and `end_date`

2. **Model Architecture** (`train.py`):
   - Customize neural network architecture in `policy_kwargs`
   - Adjust learning parameters (learning rate, batch size, etc.)
   - Modify early stopping parameters

### Performance Visualization

The system automatically generates performance plots showing:
- Portfolio value over time
- Trading action probabilities
- Individual trade annotations

## Training Process

1. **Data Preparation**:
   - Downloads historical data using yfinance
   - Calculates technical indicators
   - Normalizes features

2. **Model Training**:
   - Uses Maskable PPO for handling invalid actions
   - Implements early stopping based on validation performance
   - Saves best performing model

3. **Validation**:
   - Tests model on unseen data
   - Generates performance metrics
   - Visualizes trading decisions

## Risk Management

The system implements several risk management features:
- Maximum position size of 2% of portfolio per trade
- Maximum 10% of portfolio value per stock
- Automatic action masking for invalid trades
- 50% position reduction on sell signals

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open source and available under the MIT License.

## Disclaimer

This software is for educational purposes only. Use it at your own risk. The authors and contributors are not responsible for any financial losses incurred through the use of this system.