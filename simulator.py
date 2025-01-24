def step(self, actions):
    # Existing step logic...
    self.trade_history.append({
        'timestamp': self.current_date,
        'holdings': self.holdings.copy(),
        'balance': self.balance,
        'portfolio_value': self.portfolio_value
    })
