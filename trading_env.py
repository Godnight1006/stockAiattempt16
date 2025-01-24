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
