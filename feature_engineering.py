def add_technical_indicators(df):
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    df['RSI'] = 100 - (100 / (1 + (gain.rolling(14).mean() / loss.rolling(14).mean())))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Volatility
    df['Volatility_5'] = df['Close'].pct_change().rolling(5).std()
    df['Volatility_20'] = df['Close'].pct_change().rolling(20).std()
    
    # Momentum
    df['Momentum_5'] = df['Close'].pct_change(5)
    df['Momentum_20'] = df['Close'].pct_change(20)
    
    # Moving Averages
    for window in [20, 50, 200]:
        df[f'MA_{window}'] = df['Close'].rolling(window).mean()
    
    return df.dropna()
