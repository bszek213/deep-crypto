import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Download AAVE-USD data
list_crypt = ['BTC', 'ETH', 'ADA', 'MATIC', 'DOGE', 'SOL', 'DOT', 'SHIB', 'TRX', 'FIL', 'LINK',
                    'APE', 'MANA', "AVAX", "ZEC", "ICP", "FLOW", "EGLD", "XTZ", "LTC", "XRP", "BCH","UNI",
                    'LTC',"ETC","ATOM","AAVE","HNT","ALGO","AXS"] 
for crypt in list_crypt:
    ticker = f'{crypt}-USD'
    data = yf.download(ticker, period='max')

    # Calculate daily price change
    data['Price Change'] = data['Adj Close'].pct_change()

    # Add a column for the day of the week
    data['Day of the Week'] = data.index.day_name()

    # Determine if price went up (1) or down (0)
    data['Price Direction'] = (data['Price Change'] > 0).astype(int)

    # Group by day of the week and calculate the probability of price going up
    probabilities = data.groupby('Day of the Week')['Price Direction'].mean()

    # Reorder the days of the week to start with Sunday and end with Saturday
    ordered_days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    probabilities = probabilities.reindex(ordered_days)

    # Plot the probabilities
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=probabilities.index, y=probabilities.values, palette='coolwarm')

    # Annotate the bars with the probability values
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 10),  # Offset the text by 10 points
                    textcoords='offset points')

    # Finalize plot
    plt.title(f'Probability of {ticker} Price Going Up by Day of the Week, (n={len(data)})')
    plt.ylabel('Probability')
    plt.xlabel('Day of the Week')
    plt.ylim(0, 1)
    plt.show()
