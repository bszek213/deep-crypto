import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'

def get_price(crypt):
    crypt_name = crypt + '-USD'
    temp = yf.Ticker(crypt_name)
    if len(temp.history(period = 'max', interval="1d")) < 1:
        print(f'{crypt_name} has no data. No connection to yfinance')
        exit()
    price_data = temp.history(period = 'max', interval="1d")
    return price_data

def bin_data(df,crypt):
    df['Date'] = pd.to_datetime(df.index)
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['pct_change'] = df['Close'].pct_change()

    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    bins = pd.Categorical(df['DayOfWeek'], categories=days_of_week, ordered=True)
    grouped = df.groupby(bins)

    result = grouped.agg({
    'pct_change': ['median','std',lambda x: np.std(x, ddof=1), lambda x: t.interval(0.95, len(x)-1, loc=np.mean(x), scale=np.std(x, ddof=1)/np.sqrt(len(x)))]
    })
    median_values = result.loc[:, ('pct_change', 'median')].values
    confidence_interval = result.loc[:, ('pct_change', '<lambda_1>')]
    ci = []
    for val in confidence_interval:
        ci.append(val[1] - val[0])
    
    #plot
    plt.figure(figsize=[10,8])
    plt.errorbar(days_of_week, median_values, yerr=ci, marker='o', markersize=7,capsize=5, label='Median with 95% CI')
    plt.xlabel('Day of the Week',weight='bold')
    plt.ylabel('Pct Change',weight='bold')
    plt.title(f'Pct Change with Confidence Intervals - {crypt}',weight='bold')
    plt.legend()
    plt.tight_layout()
    if not os.path.exists('price_change'):
        os.mkdir('price_change')
    plt.savefig(os.path.join('price_change',f'{crypt}_chamge_week.png'),dpi=400)
    plt.close()

def main():
    list_crypt = ['BTC','ETH','ADA','MATIC','DOGE',
                    'SOL','DOT','SHIB','TRX','FIL','LINK',
                    'APE','MANA',"AVAX","ZEC","ICP","FLOW",
                    "EGLD","XTZ","LTC"]
    for crypt in list_crypt:
        price_data = get_price(crypt)
        bin_data(price_data,crypt)
if __name__ == "__main__":
    main()