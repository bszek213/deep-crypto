import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from tqdm import tqdm
import calendar
# from EntropyHub import PermEn, ApEn
from scipy.stats import entropy
import seaborn as sns
from scipy.optimize import curve_fit
# from sys import argv
import argparse
import requests
from bs4 import BeautifulSoup

mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'

"""
Change price and Rainbow weighted average plots
"""
def delete_png_files(directory):
    try:
        files = os.listdir(directory)
        for file in files:
            if file.endswith(".png"):
                file_path = os.path.join(directory, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_price(crypt):
    # if crypt != "^GSPC" and crypt != "VOO" and crypt != "^DJI" and crypt != "^RUT":
    #     crypt_name = crypt + '-USD'
    # else:
    #     crypt_name = crypt

    temp = yf.Ticker(crypt)
    if len(temp.history(period = 'max', interval="1d")) < 1:
        print(f'{crypt} has no data. No connection to yfinance')
        # exit()
    price_data = temp.history(period = 'max', interval="1d")
    price_data.index = pd.to_datetime(price_data.index)
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

def get_fs(df):
    time_interval = df.index[1] - df.index[0]
    time_interval_seconds = time_interval.total_seconds()
    return 1 / time_interval_seconds

def ent_causality(price):
    window_size = 300 #days
    overlap = 1  # days (you can adjust this)
    permutation_entropy_out = []
    statistical_complexity = []

    price['log_return'] = np.log(price['Close'] / price['Close'].shift(1))
    btc_returns_normalized = price['log_return'].fillna(method='bfill')
    btc_returns_normalized.replace([np.inf, -np.inf], np.nan, inplace=True)
    btc_returns_normalized.dropna(inplace=True)
    # btc_returns_normalized = (price['log_return'] - np.mean(price['log_return'])) / np.std(price['log_return'])

    # btc_returns_normalized = btc_returns_normalized.fillna(method='bfill').values

    # btc_returns_normalized = np.round(btc_returns_normalized, decimals=2)

    hist, _ = np.histogram(btc_returns_normalized, bins=50, density=True)
    #probability distribution
    probs = hist / np.sum(hist)

    #entropy and complexity
    # for i in range(0, len(btc_returns_normalized) - window_size, overlap):
    #     window_data = btc_returns_normalized[i:i+window_size]
    #     pe = PermEn(window_data)  # Permutation entropy
    #     # sc = spectral_entropy(window_data,1, method='welch')  # Statistical complexity
    #     sc = ApEn(window_data,r=0.2)
    #     permutation_entropy_out.append(pe[2][0])
    #     statistical_complexity.append(sc[0][2])
    #     print(pe[2][0])
    #     print(sc[0][2])

    # plt.figure(figsize=(8, 6))
    # plt.scatter(permutation_entropy_out, statistical_complexity, c='b', alpha=0.7)
    # plt.xlabel('Permutation Entropy')
    # plt.ylabel('Statistical Complexity')
    # plt.title('Complexity-Entropy Causality Plane for Bitcoin Price')
    # plt.grid(True)
    # plt.show()
    try:
        return entropy(probs)
    except:
        return np.nan

def logFunc(x,a,b,c):
    return a*np.log(b+x) + c

def logFunc_simple(x, a, c):
    return a * np.log(x) + c

def rainbow_plot(price,crypt):
    price['average_price'] = (price['Close'] + price['High'] + price['Low']) / 3
    price = price[price["average_price"] > 0]

    xdata = np.array([x+1 for x in range(len(price['average_price']))])
    ydata = np.log(price['average_price'])
    # try:
    try:
        popt, _ = curve_fit(logFunc, xdata, ydata, maxfev=5000)
        fittedYData = logFunc(xdata, popt[0], popt[1], popt[2])
    except:
        print('=============')
        print(f'curve fit for a*np.log(b+x) + c failed for {crypt}. Using a * np.log(x) + c instead')
        print('=============')
        popt, _ = curve_fit(logFunc_simple, xdata, ydata, maxfev=5000)
        fittedYData = logFunc_simple(xdata, popt[0], popt[1])

    #This is our fitted data
    plt.style.use("dark_background")
    plt.figure(figsize=(15,8))
    plt.semilogy(price.index.to_numpy(), price['average_price'].to_numpy(), marker='*')
    plt.title(f'{crypt} Rainbow Chart')
    plt.xlabel('Time')
    plt.ylabel(f'{crypt} price in log scale')

    for i in np.arange(-0.5, 4, 0.5):
        price[f"fitted_data{i}"] = np.exp(fittedYData + i*.455)
        # plt.plot(price.index.to_numpy(), np.exp(fittedYData + i*.455))
        plt.fill_between(price.index.to_numpy(), np.exp(fittedYData + i*.45 -1), np.exp(fittedYData + i*.45), alpha=0.4)

    if not os.path.exists('figures_rainbow'):
        os.mkdir('figures_rainbow')
    save_path = os.path.join(os.getcwd(),'figures_rainbow',f'{crypt}_rainbow.png')
    plt.savefig(save_path,dpi=400)
    plt.close()
    # except:
    #     print(f'curve_fit failed for {crypt}')

def get_trending_tickers():
    url = "https://finance.yahoo.com/trending-tickers"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        ticker_elements = soup.find_all("a", class_="Fw(600)")
        trending_tickers = [ticker.text for ticker in ticker_elements]
        return trending_tickers
    else:
        print("Failed to fetch data from Yahoo Finance.")
        return []
    
def main():
    parser = argparse.ArgumentParser(description='Process crypt names')
    parser.add_argument('--name', required=True, help='Name of the crypt. If you want all trending and cryptos type all')
    parser.add_argument('--extension', required=True, help='add -USD extension to crypto: either True or False')
    args = parser.parse_args()
    #delete old images
    delete_png_files(os.path.join(os.getcwd(),'figures_rainbow'))
    delete_png_files(os.path.join(os.getcwd(),'price_change'))
    if args.name == "all":
        #cryptos
        list_crypt = ['BTC','ETH',
                    'ADA','MATIC','DOGE',
                        'SOL','DOT','SHIB',
                        'TRX','FIL','LINK',
                        'APE','MANA',"AVAX",
                        "ZEC","ICP","FLOW",
                        "EGLD","XTZ","LTC"]
        ent = []
        for crypt in tqdm(list_crypt):
            crypt = crypt + '-USD'
            price_data = get_price(crypt)
            rainbow_plot(price_data,crypt)
            bin_data_monthly(price_data,crypt)
            # bin_data_weekly(price_data,crypt)
            ent.append(ent_causality(price_data))
        df = pd.DataFrame({'Crypto': list_crypt, 'Entropy': ent})
        df_sorted = df.sort_values(by='Entropy')
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Entropy', y='Crypto', data=df_sorted, palette='deep')
        plt.xlabel('Shannon Entropy')
        plt.ylabel('Crypto')
        plt.title('Shannon Entropy of Returns (Lowest to Highest)')
        plt.tight_layout()
        plt.savefig("Entropy_cryptos_stock_market.png",dpi=400)

        #trending
        trending_list = get_trending_tickers()
        print("Trending Tickers:", trending_list)
        trending_list = trending_list + ['^RUT','^DJI',"^GSPC","VOO"]
        for listing in tqdm(trending_list):
            try:
                price_data = get_price(listing)
                rainbow_plot(price_data,listing)
                bin_data_monthly(price_data,listing)
            except Exception as e:
                print(f'no yfinance data for {listing}: {e}')
    else:
        crypt = args.name
        if args.extension == "True":
            crypt = crypt + '-USD'
        price_data = get_price(crypt)
        rainbow_plot(price_data,crypt)
        bin_data_monthly(price_data,crypt)
        # bin_data_weekly(price_data,crypt)

 
if __name__ == "__main__":
    main()