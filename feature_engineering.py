import numpy as np

def add_technical_indicators(df):
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
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
    
    # Add normalized price
    df['Norm_Close'] = df['Close'] / df['Close'].iloc[0]
    
    # Add returns
    df['Return_1'] = df['Close'].pct_change(1)
    df['Return_5'] = df['Close'].pct_change(5)
    df['Return_20'] = df['Close'].pct_change(20)
    
    # Add volatility ratio
    df['Vol_Ratio'] = df['Volatility_5'] / df['Volatility_20']
    
    # Add momentum ratio
    df['Momentum_Ratio'] = df['Momentum_5'] / df['Momentum_20']
    
    # Add MA crossovers
    df['MA_20_50_Cross'] = (df['MA_20'] > df['MA_50']).astype(int)
    df['MA_50_200_Cross'] = (df['MA_50'] > df['MA_200']).astype(int)
    
    # Clean problematic values
    # 1. Handle infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 2. Forward fill then backward fill missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # 3. Rolling window cleanup for remaining NaNs
    cols_to_clean = ['Vol_Ratio', 'Momentum_Ratio', 'RSI', 'MACD', 'Signal']
    for col in cols_to_clean:
        df[col] = df[col].rolling(3, min_periods=1, center=True).mean()
    
    # Final cleanup for any remaining NaNs
    df.dropna(inplace=True)
    
    return df
