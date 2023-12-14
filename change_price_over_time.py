import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from tqdm import tqdm
import calendar
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

def bin_data_weekly(df,crypt):
    df['Date'] = pd.to_datetime(df.index)
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['pct_change'] = df['Close'].pct_change()

    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    bins = pd.Categorical(df['DayOfWeek'], categories=days_of_week, ordered=True)
    grouped = df.groupby(bins)

    result = grouped.agg({
    'pct_change': ['median','std', lambda x: t.interval(0.95, len(x)-1, loc=np.mean(x), scale=np.std(x, ddof=1)/np.sqrt(len(x)))]
    })
    median_values = result.loc[:, ('pct_change', 'median')].values
    confidence_interval = result.loc[:, ('pct_change', '<lambda_0>')]
    ci = []
    for val in confidence_interval:
        ci.append(val[1] - val[0])
    
    #plot
    plt.figure(figsize=[10,8])
    plt.errorbar(days_of_week, median_values, yerr=ci, marker='o', markersize=7,capsize=5, label='Median with 95% CI')
    min_index = np.argmin(median_values)
    max_index = np.argmax(median_values)
    low_month, price_low, ci_low = days_of_week[min_index], float(median_values[min_index]), float(ci[min_index])
    day_name_to_number = {day: i for i, day in enumerate(calendar.day_name) if day}
    low_day_number = day_name_to_number[low_month]
    high_day_number = day_name_to_number[days_of_week[max_index]]
    plt.errorbar(low_day_number, price_low, yerr=ci_low, 
                 marker='o', markersize=7, capsize=5, color='red', label='Lowest Median with 95% CI')
    plt.errorbar(high_day_number, median_values[max_index], yerr=ci[max_index], 
                 marker='o', markersize=7, capsize=5, color='green', label='Highest Median with 95% CI')
    plt.xlabel('Day of the Week',weight='bold')
    plt.ylabel('Pct Change',weight='bold')
    plt.title(f'Pct Change with Confidence Intervals - {crypt}',weight='bold')
    plt.legend()
    plt.tight_layout()
    if not os.path.exists('price_change'):
        os.mkdir('price_change')
    plt.savefig(os.path.join('price_change',f'{crypt}_change_week.png'),dpi=400)
    plt.close()

def bin_data_monthly(df,crypt):
    df['Date'] = pd.to_datetime(df.index)
    df['DayOfMonth'] = df['Date'].dt.month_name()
    df['pct_change'] = df['Close'].pct_change()
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July'
                    , 'August', 'September', 'October', 'November', 'December']
    bins = pd.Categorical(df['DayOfMonth'], categories=months, ordered=True)
    grouped = df.groupby(bins)
    
    result = grouped.agg({
    'pct_change': ['median','std', lambda x: t.interval(0.95, len(x)-1, loc=np.mean(x), scale=np.std(x, ddof=1)/np.sqrt(len(x)))]
    })
    median_values = result.loc[:, ('pct_change', 'median')].values
    confidence_interval = result.loc[:, ('pct_change', '<lambda_0>')]
    ci = []
    for val in confidence_interval:
        ci.append(val[1] - val[0])

    #plot
    plt.figure(figsize=[15,8])
    plt.errorbar(months, median_values, yerr=ci, marker='o', markersize=7,capsize=5, label='Median with 95% CI')
    min_index = np.argmin(median_values)
    max_index = np.argmax(median_values)
    low_month, price_low, ci_low = months[min_index], float(median_values[min_index]), float(ci[min_index])
    month_name_to_number = {month: i for i, month in enumerate(calendar.month_name) if month}
    low_month_number = month_name_to_number[low_month] - 1
    high_month_number = month_name_to_number[months[max_index]] - 1
    plt.errorbar(low_month_number, price_low, yerr=ci_low, 
                 marker='o', markersize=7, capsize=5, color='red', label='Lowest Median with 95% CI')
    plt.errorbar(high_month_number, median_values[max_index], yerr=ci[max_index], 
                 marker='o', markersize=7, capsize=5, color='green', label='Highest Median with 95% CI')
    plt.xlabel('Months',weight='bold')
    plt.ylabel('Pct Change',weight='bold')
    plt.title(f'Pct Change with Confidence Intervals - {crypt}',weight='bold')
    plt.legend()
    plt.tight_layout()
    if not os.path.exists('price_change'):
        os.mkdir('price_change')
    plt.savefig(os.path.join('price_change',f'{crypt}_change_month.png'),dpi=400)
    plt.close()


def main():
    list_crypt = ['BTC','ETH','ADA','MATIC','DOGE',
                    'SOL','DOT','SHIB','TRX','FIL','LINK',
                    'APE','MANA',"AVAX","ZEC","ICP","FLOW",
                    "EGLD","XTZ","LTC"]
    for crypt in tqdm(list_crypt):
        price_data = get_price(crypt)
        bin_data_monthly(price_data,crypt)
        bin_data_weekly(price_data,crypt)
        
if __name__ == "__main__":
    main()