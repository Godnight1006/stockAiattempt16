import matplotlib.pyplot as plt

def plot_trades(stock_data, trades):
    plt.figure(figsize=(12,6))
    plt.plot(stock_data['Close'])
    for trade in trades:
        plt.axvline(trade['date'], color='g' if trade['action'] == 'buy' else 'r', alpha=0.3)
    plt.title('Price with Trade Annotations')
    plt.show()
