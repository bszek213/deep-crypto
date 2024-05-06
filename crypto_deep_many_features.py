import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import ta
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from colorama import Fore, Style
from sys import argv
from tqdm import tqdm
from keras_tuner import RandomSearch
import yaml
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from seaborn import regplot, heatmap
from scipy.stats import pearsonr
from fredapi import Fred
import mwclient
from transformers import pipeline
from time import strftime
# from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from sampen import sampen2
import pandas_ta as ta

"""
TODO: feature engineer - add in skew, kurtosis running at different intervals - 7, 14, 30 of the close price
      save mape for each crypto over time to get a cumulative error.
      remove features that have 0 for 75% of the total array length
"""
def create_lstm_model(hp, n_steps, n_features, n_outputs):
    activation_choice = hp.Choice('activation', values=['relu', 'leaky_relu', 'tanh', 'linear'])
    regularizer_strength_l1 = hp.Float('regularizer_strength_l1', min_value=1e-6, max_value=1e-2, sampling='log')
    regularizer_strength_l2 = hp.Float('regularizer_strength_l2', min_value=1e-6, max_value=1e-2, sampling='log')
    
    regularizer_l1 = tf.keras.regularizers.l1(regularizer_strength_l1)
    regularizer_l2 = tf.keras.regularizers.l2(regularizer_strength_l2)
    
    #Batch Norm layer
    model = tf.keras.models.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hp.Int('units', min_value=1, max_value=20, step=1),
                                                           activation=activation_choice, return_sequences=False,
                                                           kernel_regularizer=regularizer_l2, recurrent_regularizer=regularizer_l1,
                                                           input_shape=(n_steps, n_features))),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(n_outputs, 
                              hp.Choice('dense_activation', values=['leaky_relu', 'tanh'])) #'linear'
    ])

    #Try Attention layer
    # inputs = tf.keras.layers.Input(shape=(n_steps, n_features))
    # lstm_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=hp.Int('units', min_value=1, max_value=20, step=1),
    #                             activation=activation_choice,
    #                             return_sequences=True,
    #                             kernel_regularizer=regularizer_l2,
    #                             recurrent_regularizer=regularizer_l1))(inputs)
    # attention_out = tf.keras.layers.Attention()([lstm_out, lstm_out])
    # outputs = tf.keras.layers.Dense(n_outputs, activation=hp.Choice('dense_activation', values=['linear','leaky_relu', 'tanh']))(attention_out)
    # model = Model(inputs=inputs, outputs=outputs)

    optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop']) #, 'sgd'
    if optimizer_choice == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Float('adam_learning_rate', min_value=0.0001, max_value=0.01, sampling='log'))
    else:# optimizer_choice == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp.Float('rmsprop_learning_rate', min_value=0.0001, max_value=0.01, sampling='log'))
    # else:
    #     optimizer = tf.keras.optimizers.SGD(learning_rate=hp.Float('sgd_learning_rate', min_value=0.0001, max_value=0.001, sampling='log'))
    
    model.compile(optimizer=optimizer,
                  loss='mean_absolute_error', #mean_squared_error
                  metrics=['mean_absolute_error'])

    return model

def load_data_from_yaml(filename):
    try:
        with open(filename, 'r') as file:
            data = yaml.safe_load(file)
            if data:
                return data
            else:
                return {}
    except FileNotFoundError:
        return {}
    
def update_crypto_data(filename, crypto_name, price, error):
    crypto_data = load_data_from_yaml(filename)
    crypto_data[crypto_name] = {'price': [float(x) for x in price], 'error': [float(error)], "time": [datetime.now()]}
    with open(filename, 'w') as file:
        yaml.dump(crypto_data, file)

def find_crypto_data(filename, crypto_name):
    crypto_data = load_data_from_yaml(filename)
    return crypto_data.get(crypto_name)
    
def calculate_classic_pivot_points_func(high, low, close):
    """
    Pivot points indicates direction - price above the pivot point, "May" go up. vice versa
    Support and resistance are good for stop losses and exit points
    """
    pivot_point = (high + low + close) / 3.0
    support1 = (2 * pivot_point) - high
    resistance1 = (2 * pivot_point) - low
    support2 = pivot_point - (high - low)
    resistance2 = pivot_point + (high - low)
    support3 = low - 2 * (high - pivot_point)
    resistance3 = high + 2 * (pivot_point - low)
    return pivot_point, support1, resistance1, support2, resistance2, support3, resistance3

def calculate_classic_pivot_points(df):
    df['pivot'], df['s1'], df['r1'], df['s2'], df['r2'], df['s3'], df['r3'] = zip(
        *df.apply(lambda row: calculate_classic_pivot_points_func(row['High'], row['Low'], row['Close']), axis=1)
    )
    return df

def calculate_mean(sentiments):
    if isinstance(sentiments, list):
        return np.mean(sentiments)
    else:
        return sentiments
    
def feature_engineer(data):
    for col in data.columns:
        if data[col].notna().all():
            #create running averages
            data[f'{col}_short'] = data[col].rolling(window=7,min_periods=1).mean()
            data[f'{col}_med'] = data[col].rolling(window=31,min_periods=1).mean()
            data[f'{col}_long'] = data[col].rolling(window=90,min_periods=1).mean()
            # #skewness
            # data[f'{col}_short_skew'] = data[col].rolling(window=7,min_periods=1).skew()
            # data[f'{col}_med_skew'] = data[col].rolling(window=31,min_periods=1).skew()
            # data[f'{col}_long_skew'] = data[col].rolling(window=90,min_periods=1).skew()
            # #kurt
            # data[f'{col}_short_kurt'] = data[col].rolling(window=7,min_periods=1).kurt()
            # data[f'{col}_med_kurt'] = data[col].rolling(window=31,min_periods=1).kurt()
            # data[f'{col}_long_kurt'] = data[col].rolling(window=90,min_periods=1).kurt()
            #variance
            # data[f'{col}_short_var'] = data[col].rolling(window=7,min_periods=1).var()
            # data[f'{col}_med_kurt_var'] = data[col].rolling(window=31,min_periods=1).var()
            # data[f'{col}_long_kurt_var'] = data[col].rolling(window=90,min_periods=1).var()
    #add day of the week and month
    # data['day_of_week'] = data.index.weekday
    # data['month'] = data.index.month
    return data

def box_count(data, box_size):
    """
    Count the number of boxes needed to cover the data at a given box size.
    """
    num_boxes = len(data) // box_size
    return num_boxes

def calculate_fractal_dimension(data):
    """
    Calculate the fractal dimension using the box-counting method.
    """
    box_sizes = np.logspace(0.1, 3, 100, dtype=int)
    box_counts = []

    for box_size in box_sizes:
        num_boxes = box_count(data, box_size)
        box_counts.append(num_boxes)

    log_box_sizes = np.log(box_sizes)
    log_box_counts = np.log(box_counts)
    slope, _ = np.polyfit(log_box_sizes, log_box_counts, 1)

    return slope

class changePricePredictor:
    def __init__(self, crypt, n_features, n_steps, n_outputs, n_epochs, batch_size):
        self.which_analysis = 'pca' #pca or corr
        self.crypt_name = crypt
        # self.n_features = n_features
        self.n_steps = n_steps
        self.n_outputs = n_outputs
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        #use S&P price for any explanatory power of what the crypto will do
        # temp = yf.Ticker('^GSPC')
        # sp_price = temp.history(period = 'max', interval="1d")
        # self.typical_price_SP = (sp_price['High'] + sp_price['Close'] + sp_price['Low'])
        # self.typical_price_SP.name = 'typical_price_SP'
        # self.typical_price_SP.index = self.typical_price_SP.index.date
        # #same thing for usd 
        # usd_eur_data = yf.download('USDEUR=X')
        # self.typical_price_usd = (usd_eur_data['High'] + usd_eur_data['Close'] + usd_eur_data['Low'])
        # self.typical_price_usd.name = 'typical_price_usd_euro'
        # self.typical_price_usd.index = self.typical_price_usd.index.date

        if self.crypt_name == "SP":
            crypt_name = "SP"
            temp = yf.Ticker('^GSPC')
            price_data = temp.history(period = 'max', interval="1d")
        else:
            crypt_name = self.crypt_name + "-USD"
            temp = yf.Ticker(crypt_name)
            if len(temp.history(period = 'max', interval="1d")) < 1:
                print(f'{crypt_name} has no data. No connection to yfinance')
                exit()
            price_data = temp.history(period = 'max', interval="1d")
        #change the date out of utc to mine
        current_date = datetime.now().date()
        new_index = pd.date_range(end=current_date, periods=len(price_data.index), freq='D')
        price_data.index = new_index
        
        print(Fore.GREEN,f'NUMBER OF SAMPLES FOR {crypt_name}: {len(price_data)}',Style.RESET_ALL)
        # self.features = ['Close','Open', 'High', 'Low','Volume', 'Dividends', 'Stock Splits',
        #                             'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em',
        #                             'volume_sma_em', 'volume_vpt', 'volume_vwap', 'volume_mfi',
        #                             'volume_nvi', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
        #                             'volatility_bbw', 'volatility_bbp', 'volatility_bbhi',
        #                             'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl',
        #                             'volatility_kcw', 'volatility_kcp', 'volatility_kchi',
        #                             'volatility_kcli', 'volatility_dcl', 'volatility_dch', 'volatility_dcm',
        #                             'volatility_dcw', 'volatility_dcp', 'volatility_atr', 'volatility_ui',
        #                             'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
        #                             'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
        #                             'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_ind_diff',
        #                             'trend_trix', 'trend_mass_index', 'trend_dpo', 'trend_kst',
        #                             'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
        #                             'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',
        #                             'trend_stc', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci',
        #                             'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',
        #                             'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up',
        #                             'trend_psar_down', 'trend_psar_up_indicator',
        #                             'trend_psar_down_indicator', 'momentum_rsi', 'momentum_stoch_rsi',
        #                             'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'momentum_tsi',
        #                             'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',
        #                             'momentum_ao', 'momentum_roc', 'momentum_ppo', 'momentum_ppo_signal',
        #                             'momentum_ppo_hist', 'momentum_pvo', 'momentum_pvo_signal',
        #                             'momentum_pvo_hist', 'momentum_kama', 'others_dr', 'others_dlr',
        #                             'others_cr']
        # self.non_close_features = ['Volume','Open', 'High', 'Low',
        #                             'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em',
        #                             'volume_sma_em', 'volume_vpt', 'volume_vwap', 'volume_mfi',
        #                             'volume_nvi', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
        #                             'volatility_bbw', 'volatility_bbp', 'volatility_bbhi',
        #                             'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl',
        #                             'volatility_kcw', 'volatility_kcp', 'volatility_kchi',
        #                             'volatility_kcli', 'volatility_dcl', 'volatility_dch', 'volatility_dcm',
        #                             'volatility_dcw', 'volatility_dcp', 'volatility_atr', 'volatility_ui',
        #                             'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
        #                             'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
        #                             'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_ind_diff',
        #                             'trend_trix', 'trend_mass_index', 'trend_dpo', 'trend_kst',
        #                             'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
        #                             'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',
        #                             'trend_stc', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci',
        #                             'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',
        #                             'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up',
        #                             'trend_psar_down', 'trend_psar_up_indicator',
        #                             'trend_psar_down_indicator', 'momentum_rsi', 'momentum_stoch_rsi',
        #                             'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'momentum_tsi',
        #                             'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',
        #                             'momentum_ao', 'momentum_roc', 'momentum_ppo', 'momentum_ppo_signal',
        #                             'momentum_ppo_hist', 'momentum_pvo', 'momentum_pvo_signal',
        #                             'momentum_pvo_hist', 'momentum_kama', 'others_dr', 'others_dlr',
        #                             'others_cr'] #'Open', 'High', 'Low', 'Dividends', 'Stock Splits',

        #pandas ta
        price_data = price_data.rename_axis("datetime")
        price_data.drop(labels=['Dividends', 'Stock Splits'], axis=1, inplace=True)
        function_names = [
            "aberration", "above", "above_value", "accbands", "ad", "adosc", "adx", "alma", "amat", "ao", "aobv", "apo",
            "aroon", "atr", "bbands", "below", "below_value", "bias", "bop", "brar", "cci", "cdl_pattern", "cdl_z", "cfo",
            "cg", "chop", "cksp", "cmf", "cmo", "coppock", "cross", "cross_value", "cti", "decay", "decreasing", "dema",
            "dm", "donchian", "dpo", "ebsw", "efi", "ema", "entropy", "eom", "er", "eri", "fisher", "fwma", "ha", "hilo",
            "hl2", "hlc3", "hma", "hwc", "hwma", "ichimoku", "increasing", "inertia", "jma", "kama", "kc", "kdj", "kst",
            "kurtosis", "kvo", "linreg", "log_return", "long_run", "macd", "mad", "massi", "mcgd", "median", "mfi",
            "midpoint", "midprice", "mom", "natr", "nvi", "obv", "ohlc4", "pdist", "percent_return", "pgo", "ppo", "psar",
            "psl", "pvi", "pvo", "pvol", "pvr", "pvt", "pwma", "qqe", "qstick", "quantile", "rma", "roc", "rsi", "rsx",
            "rvgi", "rvi", "short_run", "sinwma", "skew", "slope", "sma", "smi", "squeeze", "squeeze_pro", "ssf", "stc",
            "stdev", "stoch", "stochrsi", "supertrend", "swma", "t3", "td_seq", "tema", "thermo", "tos_stdevall", "trima",
            "trix", "true_range", "tsi", "tsignals", "ttm_trend", "ui", "uo", "variance", "vhf", "vidya", "vortex", "vp",
            "vwap", "vwma", "wcp", "willr", "wma", "xsignals", "zlma", "zscore"
        ]

        functions = {name: getattr(price_data.ta, name) for name in function_names}
        for name, func in tqdm(functions.items()):
            print(f'{name} indicator')
            try:
                func(append=True)
            except:
                print(f'{name} indicator did not work')
        #fill in nans
        threshold = len(price_data) * 0.25
        price_data = price_data.dropna(thresh=threshold, axis=1) #I just learned this
        price_data = price_data.interpolate(method='linear', limit_direction='both')
        self.data = price_data
        #get features for non-close and close
        self.features = self.data.columns.tolist()
        self.non_close_features = self.features.copy()
        self.non_close_features.remove('Close')
        self.n_features = len(self.non_close_features)
        # self.data = ta.add_all_ta_features(
        #     price_data,
        #     open="Open",
        #     high="High",
        #     close='Close',
        #     low='Low',
        #     volume='Volume',
        #     fillna=True
        # )
         
        #calc pivot points
        df_weekly = self.data.resample('M').mean()
        df = calculate_classic_pivot_points(df_weekly)
        # Plot the original data
        last_month_start = pd.Timestamp.now() - pd.DateOffset(months=2)
        last_month_data = self.data[self.data.index >= last_month_start]
        last_month_data['Close'].plot(figsize=(12, 8), label='Close Price',marker='o')

        # Plot horizontal lines for Support 1, Support 2, Resistance 1, and Resistance 2
        plt.axhline(y=df['pivot'].iloc[-1], color='purple', linestyle='--', label='Pivot Point')
        plt.axhline(y=df['s1'].iloc[-1], color='g', linestyle='--', label='Support 1')
        plt.axhline(y=df['s2'].iloc[-1], color='b', linestyle='--', label='Support 2')
        plt.axhline(y=df['r1'].iloc[-1], color='r', linestyle='--', label='Resistance 1')
        plt.axhline(y=df['r2'].iloc[-1], color='orange', linestyle='--', label='Resistance 2')
        plt.axhline(y=df['r3'].iloc[-1], color='tab:cyan', linestyle='--', label='Resistance 3')
        plt.axhline(y=df['s3'].iloc[-1], color='tab:brown', linestyle='--', label='Support 3')

        plt.title(f'{self.crypt_name} Current Pivot points - Plot Last 2 Months')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        if not os.path.exists('figures_support_resistance'):
            os.mkdir('figures_support_resistance')
        save_path = os.path.join(os.getcwd(),'figures_support_resistance',f'{self.crypt_name}_supp_resist.png')
        plt.savefig(save_path,dpi=350)
        plt.close()

    def fred_data(self):
        with open('fred_api.txt', 'r') as f:
            fred_api = f.read()
        fred = Fred(api_key=fred_api)
        #GDP 
        gdp_df = fred.get_series_as_of_date('GDP', str(datetime.now().date()))
        gdp_df['date'] = pd.to_datetime(gdp_df['date'])
        closest_date = min(gdp_df['date'], key=lambda x: abs(x - self.data.index[0]))
        desired_index = gdp_df[gdp_df['date'] == closest_date].index[0]
        gdp_df = gdp_df[['date','value']].iloc[desired_index:]
        interpolated_gdp_values = np.interp(
            np.linspace(0, 1, num=len(self.data)),
            np.linspace(0, 1, num=len(gdp_df['value'])),  # Current indices
            gdp_df['value'].fillna(method='ffill').fillna(method='bfill')  # Fill missing values and perform interpolation
        )   
        
        #inflation
        inflation_df = fred.get_series_as_of_date('T10YIEM', str(datetime.now().date()))  
        inflation_df['date'] = pd.to_datetime(inflation_df['date'])
        closest_date = min(inflation_df['date'], key=lambda x: abs(x - self.data.index[0]))
        desired_index = inflation_df[inflation_df['date'] == closest_date].index[0]
        inflation_df = inflation_df[['date','value']].iloc[desired_index:]
        interpolated_inflation_values = np.interp(
            np.linspace(0, 1, num=len(self.data)), 
            np.linspace(0, 1, num=len(inflation_df['value'])),  # Current indices
            inflation_df['value'].fillna(method='ffill').fillna(method='bfill')  # Fill missing values and perform interpolation
        )   

        #mortgage
        morg_df = fred.get_series_as_of_date('MORTGAGE30US', str(datetime.now().date()))  
        morg_df['date'] = pd.to_datetime(morg_df['date'])
        closest_date = min(morg_df['date'], key=lambda x: abs(x - self.data.index[0]))
        desired_index = morg_df[morg_df['date'] == closest_date].index[0]
        morg_df = morg_df[['date','value']].iloc[desired_index:]
        interpolated_morg_values = np.interp(
            np.linspace(0, 1, num=len(self.data)), 
            np.linspace(0, 1, num=len(morg_df['value'])),  # Current indices
            morg_df['value'].fillna(method='ffill').fillna(method='bfill')  # Fill missing values and perform interpolation
        )
        #add these to data and add feature names
        self.data['gdp'] = interpolated_gdp_values
        self.data['inflation'] = interpolated_inflation_values
        self.data['mortgage'] = interpolated_morg_values
        self.features = self.features + ['gdp','inflation','mortgage']
        self.non_close_features = self.non_close_features + ['gdp','inflation','mortgage']

    def mw_data(self):
        """
        idea from https://www.youtube.com/watch?v=TF2Nx_ifmrU
        """
        list_crypt = ['BTC','ETH','ADA','MATIC','DOGE',
                    'SOL','DOT','SHIB','TRX','FIL','LINK',
                    'APE','MANA',"AVAX","ZEC","ICP","FLOW",
                    "EGLD","XTZ","LTC","XRP"] 
        site = mwclient.Site("en.wikipedia.org")
        #handle names
        if self.crypt_name == "BTC":
            name = "Bitcoin"
        elif self.crypt_name == "ETH":
            name = "Ethereum"
        elif self.crypt_name == "ADA":
            name = "Cardano (blockchain platform)"
        elif self.crypt_name == "MATIC":
            name = "Polygon (blockchain)"
        elif self.crypt_name == "DOGE":
            name = "Dogecoin"
        elif self.crypt_name == "SOL":
            name = "Solana (blockchain platform)"
        elif self.crypt_name == "DOT":
            name = "Polkadot (cryptocurrency)"
        elif self.crypt_name == "SHIB":
            name = "Shiba Inu (cryptocurrency)"
        elif self.crypt_name == "TRX":
            name = "Tron (cryptocurrency)"
        elif self.crypt_name == "FIL":
            name = "Filecoin"
        elif self.crypt_name == "LINK":
            name = "Chainlink (blockchain)"
        elif self.crypt_name == "APE":
            name = "Bored Ape"
        elif self.crypt_name == "MANA":
            name = "Decentraland"
        elif self.crypt_name == "AVAX":
            name = "Avalanche (blockchain platform)"
        elif self.crypt_name == "ZEC":
            name = "Zcash"
        elif self.crypt_name == "ICP": #does not have one
            name = "Nothing"
        elif self.crypt_name == "FLOW":
            name = "Flow Traders"
        elif self.crypt_name == "EGLD":
            name = "Nothing"
        elif self.crypt_name == "XTZ":
            name = "Tezos"
        elif self.crypt_name == "LTC":
            name = "Litecoin"
        elif self.crypt_name == "XRP":
            name = "Ripple (payment protocol)"

        if name != "Nothing":
            page = site.pages[name]
            revs = list(page.revisions())
            #load sentiment pipeline
            sentiment_pip = pipeline("sentiment-analysis")
            #sort in reverse order
            revs = sorted(revs, key=lambda rev: rev['timestamp'])
            edits = {}
            for rev in tqdm(revs):
                if 'comment' in rev:
                    date = strftime("%Y-%m-%d",rev['timestamp'])
                    if date not in edits:
                        edits[date] = dict(sentiments=list(),edit_count=0)
                    edits[date]['edit_count'] +=1
                    sent = sentiment_pip(rev['comment'])[0]
                    if sent['label'] == "NEGATIVE":
                        sent['score'] *= -1
                    edits[date]['sentiments'].append(np.mean(sent['score']))

            df = pd.DataFrame.from_dict(edits,orient="index")
            df.index = pd.to_datetime(df.index)
            #fill in missing days
            dates_all = pd.date_range(start=next(iter(edits)),end=datetime.today())
            df = df.reindex(dates_all,fill_value=0)
            df['sentiments'] = df['sentiments'].apply(lambda x: calculate_mean(x))
            # plt.plot(df.index.to_numpy(),df['edit_count'].to_numpy())
            # plt.hist(df['sentiments'].to_numpy(),bins=75)
            # plt.show()
            #add these to data and add feature names
            # print(self.data)
            # print(df)
            len_t = len(self.data)
            self.data = pd.merge(self.data, df, left_index=True, right_index=True, how='outer')
            self.data = self.data.fillna(0)
            self.data = self.data.tail(len_t)
            #write the feature names to your saved lists
            self.features = self.features + ['sentiments','edit_count']
            self.non_close_features = self.non_close_features + ['sentiments','edit_count']

    def prepare_data(self, data):
        #FRED features
        self.fred_data()
        self.mw_data()

        #identify inf values
        is_inf = self.data.isin([np.inf, -np.inf])
        self.data[is_inf] = 0

        #Extract relevant features
        data = self.data[self.features]

        #Scale data
        # self.scaler2 = MinMaxScaler(feature_range=(0, 1))
        self.scaler2 = StandardScaler()
        self.pca = PCA(n_components=0.95)
        # self.scaler1 = MinMaxScaler(feature_range=(0, 1))
        # self.scaler1 = StandardScaler()
        
        #Close price
        data_close = data['Close'].pct_change().fillna(0).to_numpy().reshape(-1, 1) #close price
        # plt.hist(data_close,bins=400)
        # plt.show()
        # data_close = self.scaler1.fit_transform(data_close)

        data_non_close = data[self.non_close_features]
        
        #feature engineer
        # data_non_close = feature_engineer(data_non_close)

        #put sp typical price into df
        # data_non_close.index = pd.to_datetime(data_non_close.index)
        # data_non_close['Date'] = data_non_close.index.date
        # data_non_close = pd.merge(data_non_close, self.typical_price_SP, left_on='Date', right_index=True, how='left')
        # data_non_close.drop('Date', axis=1, inplace=True)
        # #sp no weekend data
        # data_non_close['typical_price_SP'].fillna(method='bfill', inplace=True)
        # #Same thing for usdeuro
        # data_non_close['Date'] = data_non_close.index.date
        # data_non_close = pd.merge(data_non_close, self.typical_price_usd, left_on='Date', right_index=True, how='left')
        # data_non_close.drop('Date', axis=1, inplace=True)
        # #eurousd no weekend data
        # data_non_close['typical_price_usd_euro'].fillna(method='bfill', inplace=True)

        #Remove correlated features
        if self.which_analysis == 'corr':
            threshold = 0.95
            correlation_matrix = data_non_close.corr()
            mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            to_drop = [column for column in correlation_matrix.columns if any(correlation_matrix.loc[column, mask[:, correlation_matrix.columns.get_loc(column)]] > threshold)]
            print(Fore.LIGHTCYAN_EX, Style.BRIGHT,f'Features to be removed: {to_drop}',Style.RESET_ALL)
            self.drop_features = to_drop
            #features to keep
            self.non_close_features = [item for item in self.non_close_features if item not in to_drop]
            # Remove highly correlated features
            data_non_close = data_non_close.drop(columns=to_drop)
            # Create and save a heatmap plot
            plt.figure(figsize=(20, 20))
            heatmap(correlation_matrix, cmap="coolwarm", mask=mask)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.savefig("correlation_heatmap.png")
            plt.close()
            data_non_close = self.scaler2.fit_transform(data_non_close)
            self.data_non_close_save = data_non_close
            data = np.concatenate((data_close, data_non_close), axis=1)
        else:
            data_non_close = self.scaler2.fit_transform(data_non_close)
            print(f'Number of features before PCA: {data_non_close.shape[1]}')
            self.data_non_close_save = self.pca.fit_transform(data_non_close)
            print(f'Number of features after PCA: {self.data_non_close_save.shape[1]}')
            plt.figure()
            plt.figure(figsize=(8, 6))
            plt.bar(range(self.pca.n_components_), self.pca.explained_variance_ratio_)
            plt.xlabel('Principal Component',fontweight='bold')
            plt.ylabel('Explained Variance Ratio',fontweight='bold')
            plt.title(f'Explained Variance Ratio of Principal Components - {self.crypt_name}',fontweight='bold')
            if not os.path.exists('pca_plots'):
                os.mkdir('pca_plots')
            plt.savefig(os.path.join(os.getcwd(),'pca_plots',f'{self.crypt_name}_pca_components.png'),dpi=400)
            plt.close()
            data = np.concatenate((data_close, self.data_non_close_save), axis=1)

        # data_sorted = np.sort(data[:,0])
        # q1, q3 = np.percentile(data_sorted, [25, 75])
        # iqr_value = q3 - q1

        # # Define the lower and upper bounds for outliers
        # lower_bound = q1 - 1.5 * iqr_value
        # upper_bound = q3 + 1.5 * iqr_value

        # # Identify outliers
        # outliers = (data_sorted < lower_bound) | (data_sorted > upper_bound)

        # # Calculate the proportion of outliers
        # proportion_outliers = np.sum(outliers) / len(data)

        # print(proportion_outliers)
        # input()

        # Split data into input/output sequences
        X, y = [], []
        for i in range(len(data)-self.n_steps-self.n_outputs+1):
            X.append(data[i:i+self.n_steps, :])
            y.append(data[i+self.n_steps:i+self.n_steps+self.n_outputs, 0])
        X, y = np.array(X), np.array(y)

        # Split data into training/validation sets
        split_idx_train = int(len(X) * 0.8)
        split_idx_val = int(len(X) * 0.9)

        X_train, y_train = X[:split_idx_train], y[:split_idx_train]
        X_val, y_val = X[split_idx_train:split_idx_val], y[split_idx_train:split_idx_val]
        X_test, y_test = X[split_idx_val:], y[split_idx_val:]

        return X_train, y_train, X_val, y_val, X_test, y_test

    # def create_model(self):
    #     # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     #     initial_learning_rate=0.01,
    #     #     decay_steps=1000,
    #     #     decay_rate=0.9,
    #     #     staircase=True
    #     # )
    #     # drop_val = 0.3
    #     model = tf.keras.models.Sequential([
    #         tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, activation='leaky_relu',return_sequences=False, 
    #                                                            input_shape=(self.n_steps, self.n_features))),
    #         tf.keras.layers.BatchNormalization(),
    #         # tf.keras.layers.Dropout(drop_val),
    #         # tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(self.n_outputs,activation="linear")
    #     ])
    #     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss='mean_squared_error',
    #                   metrics=[tf.keras.metrics.RootMeanSquaredError()])
    #     return model

    def train_model(self, X_train, y_train, X_val, y_val,X_test,y_test):
        save_path = os.path.join(os.getcwd(),'model_loc')
        if os.path.isdir(save_path):
            print('path exists')
        else:
            os.mkdir('model_loc')
        save_path = os.path.join(save_path,f"{self.crypt_name}_lstm_model.h5")
        if os.path.exists(save_path):
            self.model = load_model(save_path)

            # masker = shap.maskers.Independent(data=X_train)
            # explainer = shap.Explainer(self.model, masker)
            # shap_values = explainer.shap_values(X_train)
            # feature_importances = np.mean(np.abs(shap_values),axis=0)
            # print(feature_importances.shape)
            # feature_names = self.non_close_features
            # shap.summary_plot(feature_importances.T, 
            #                 feature_names=feature_names, 
            #                 plot_type="bar", 
            #                 max_display=feature_importances.shape[0],
            #                 show=True)
        else:
            # model = self.create_model()
            #TUNE LSTM
            tuner = RandomSearch(
                lambda hp: create_lstm_model(hp, self.n_steps, len(self.features), self.n_outputs),
                objective='val_mean_absolute_error', #val_loss
                max_trials=30,
                directory=f'{self.crypt_name}_lstm_hp',
                project_name='lstm_hyperparameter_tuning',
                # overwrite=True
            )
            #val_mean_absolute_error or val_loss
            early_stopping = EarlyStopping(monitor='val_mean_absolute_error', patience=9, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
            tuner.search(x=X_train, y=y_train,
                    epochs=200,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping,reduce_lr])
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            self.best_model = tuner.get_best_models(num_models=1)[0]
            #fit tuned model
            init_model_acc = float(10000000.0)
            for i in tqdm(range(10)):
                self.best_model.fit(X_train, y_train, epochs=200, 
                                    validation_data=(X_val, y_val),
                                    callbacks=[early_stopping,reduce_lr])
                loss, mse = self.best_model.evaluate(X_test, y_test)
                if mse < init_model_acc:
                    #save model
                    self.model = self.best_model
                    init_model_acc = mse
            print('====================================================================================')
            print(Fore.GREEN, Style.BRIGHT,f'LOWEST MSE ON TEST DATA: {init_model_acc}',Style.RESET_ALL)
            print('====================================================================================')
            #write best hyperparameters to file
            file_path = 'best_hp.txt'
            content_to_append = f"{self.crypt_name} Best Hyperparameters: {best_hps.values}"
            with open(file_path, 'a') as file:
                file.write(content_to_append + '\n')
            
            save_path = os.path.join(os.getcwd(),'model_loc')
            if os.path.isdir(save_path):
                print('path exists')
            else:
                os.mkdir('model_loc')
            save_path = os.path.join(save_path,f"{self.crypt_name}_lstm_model.h5")
            self.best_model.save(save_path)

    def evaluate_model(self, X_test, y_test):
        loss = self.model.evaluate(X_test, y_test)
        print(Fore.YELLOW, f"Test Loss: {loss}",Style.RESET_ALL)
        print(self.features)
        self.loss_output = round(loss[0],4)
        return round(loss[0],4)

    def predict(self, data):
        #save data for test
        # test = data['Close'].pct_change().fillna(method='bfill').to_numpy().reshape(-1, 1)[-self.n_steps:] #pct_change
        test = data['Close'].pct_change().to_numpy().reshape(-1, 1)[-self.n_steps:] #close
        # Prepare data for prediction
        # data_close = data['Close'].pct_change().fillna(method='bfill').to_numpy().reshape(-1, 1) #pct_change
        data_close = data['Close'].pct_change().to_numpy().reshape(-1, 1) #close
        # data_close = self.scaler1.transform(data_close)
        # data_non_close = data[self.non_close_features]
        #feature engineer
        # data = data[self.non_close_features]
        # data = feature_engineer(data)
        # data_non_close= data.drop(columns=self.drop_features)
        #transform
        # data_non_close = self.scaler2.transform()
        data = np.concatenate((data_close, self.data_non_close_save), axis=1)

        #Predict the future after test data
        if argv[2] == "test_future":
            # Make prediction on test data
            X_pred = np.array([data[-self.n_steps*2:-self.n_steps, :]])
            #pct-change
            # y_pred = self.model.predict(X_pred)
            # y_pred = y_pred.flatten()
            #close
            y_pred = self.model.predict(X_pred)
            y_pred = self.scaler1.inverse_transform(y_pred)[0]

            print(Fore.RED,y_pred,Style.RESET_ALL)
            print(Fore.GREEN,test.flatten(),Style.RESET_ALL)

            #check accuracy of prediction   
            correct = 0
            incorrect = 0  
            test_pct = np.diff(test.flatten())
            y_pred_pct = np.diff(y_pred)
            for test_val, pred_val in zip(test_pct,y_pred_pct):
                if test_val < 0 and pred_val < 0:
                    correct += 1
                elif test_val > 0 and pred_val > 0:
                    correct += 1
                elif test_val < 0 and pred_val > 0:
                    incorrect += 1
                elif test_val > 0 and pred_val < 0:
                    incorrect += 1
            print('=======================================')
            print(Fore.YELLOW, f'MAPE test data: {round(mean_absolute_percentage_error(test.flatten(),y_pred)*100,2)} %',Style.RESET_ALL)
            print(Fore.YELLOW, f'RMSE test data: {round(mean_squared_error(test.flatten(),y_pred,squared=False),10)}',Style.RESET_ALL)
            print(Fore.YELLOW, "R2 score test data:", r2_score(test.flatten(),y_pred),Style.RESET_ALL)
            print('=======================================')
            print(Fore.GREEN,f'correct direction: {correct / (correct + incorrect)}',Style.RESET_ALL,
                Fore.RED,f'incorrect direction: {incorrect / (correct + incorrect)}',Style.RESET_ALL)
            print('=======================================')
            plt.plot(y_pred,color='r',marker='*',alpha=0.3,label='pred')
            plt.plot(test.flatten(),marker='*',color='g',alpha=0.3,label='test')
            plt.legend()
            plt.show()
            # # Prepare data for prediction
            # data = data[['Close', 'Low', 'High', 'MACD', 'RSI']]
            # data = self.scaler.transform(data)
            # X_pred = np.array([data[-self.n_steps:, :]])
            # X_pred = np.reshape(X_pred, (1, self.n_steps, self.n_features))
            # # Make prediction
            # y_pred = self.model.predict(X_pred)
            # print(y_pred)
            # y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, self.n_features))[0]  
            # # y_pred = y_pred[:, 0] # extract only the predicted Close prices
            return y_pred, self.data.index[-1], test
        else:
            X_pred = np.array([data[-self.n_steps:, :]])
            y_pred = self.model.predict(X_pred)
            # y_pred = self.scaler1.inverse_transform(y_pred)[0]
            self.y_pred = y_pred.flatten()
            #tomorrow price 
            print(Fore.GREEN, Style.BRIGHT, f'Predicted value for tomorrow at {datetime.now() + timedelta(days=1)} for {self.crypt_name}: {y_pred[0][0]*100}%',Style.RESET_ALL)

            return y_pred, self.data.index[-1], 0

    def rolling_95_pct_ci(self, close_data):
        window_size = 14
        rolling_std = close_data.rolling(window=window_size).std()
        #z-score for a 95% confidence interval in a standard normal distribution
        z_score = 1.96  
        ci_upper = close_data + (z_score * rolling_std)
        ci_lower = close_data - (z_score * rolling_std)
        return ci_upper, ci_lower

    def plot_pct_change(self):
        close_data = self.data['Close']
        look_back = 30 #days
        # last_week = self.data['Close'].iloc[-look_back:]
        # look_back_confidence_interval_upper = ci_upper.iloc[-look_back:]
        # look_back_confidence_interval_lower = ci_lower.iloc[-look_back:]
        price_val = close_data.iloc[-1]
        future_close = pd.Series(close_data.values,index=close_data.index)
        for val in self.y_pred:
            tomorrow = price_val + (price_val * val)
            price_val = tomorrow
            future_close[future_close.index[-1] + pd.Timedelta(days=1)] = tomorrow
        # calculate 95% CI rolling
        ci_upper, ci_lower = self.rolling_95_pct_ci(future_close)
        future_close.index = pd.to_datetime(future_close.index)

        plt.rcParams['font.weight'] = 'bold'
        plt.figure(figsize=(12,10))
        look_back_2 = look_back# + len(self.y_pred)
        plt.fill_between(ci_upper.index.to_numpy()[-look_back_2:], ci_upper.to_numpy()[-look_back_2:], 
                         ci_lower.to_numpy()[-look_back_2:], 
                         color='lightblue', alpha=0.5, label='rolling 95% ci')
        plt.plot(future_close.index.to_numpy()[-look_back:], 
                 future_close.to_numpy()[-look_back:], color='blue', marker='o', label='Price')
        plt.plot(future_close.index.to_numpy()[-len(self.y_pred):], future_close.to_numpy()[-len(self.y_pred):], color='green', marker='o', label='Predicted Price')
        # plt.plot(last_week.index[-2:].to_numpy(), last_week[-2:].to_numpy(), color='blue')  # Connect the last two points with a blue line
        plt.xlabel('Date',fontweight='bold')
        plt.ylabel('Price',fontweight='bold')
        plt.title(f'{self.crypt_name} price for the next {len((self.y_pred))} days',fontweight='bold')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        if not os.path.exists('figures'):
            os.mkdir('figures')
        save_path = os.path.join(os.getcwd(),'figures',f'{self.crypt_name}_future_price.png')
        plt.savefig(save_path,dpi=350)
        plt.close()

    def plot_results(self):
        pred = pd.read_csv(f'{self.crypt_name}_pred.csv')
        # plt.plot(self.data['Close'], label='Actual')
        plt.plot(pred['date'],pred['pred'], marker='*',label='Predicted')
        plt.title(f'{self.crypt_name} Close Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Close Price (USD)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

    def save_previous_days_data(self):
        filename = 'crypto_pre_error.yaml'
        save_price = [] 
        price_val = self.data['Close'].iloc[-1:].values
        for val in self.y_pred:
            tomorrow = price_val + (price_val * val)
            price_val = tomorrow
            save_price.append(tomorrow) 
        # tomorrow = self.data['Close'].iloc[-1:].values + (self.data['Close'].iloc[-1:].values * self.y_pred)
        update_crypto_data(filename, self.crypt_name, save_price, self.loss_output)
        if os.path.exists('predictions'):
            os.mkdir('predictions')
        # np.savetxt(os.path.join(os.getcwd(),'predictions',f'{self.crypt_name}_prediction.txt'), self.y_pred[0][0], fmt='%.6f')

    def check_output(self):
        # #calculate 95% CI rolling
        # self.rolling_95_pct_ci()

        with open('crypto_pre_error.yaml', 'r') as file:
            data = yaml.safe_load(file)
        time_output = [data[self.crypt_name]['time'][0] + timedelta(days=i) for i in range(1, len(data[self.crypt_name]['price']))]
        time_output.insert(0, data[self.crypt_name]['time'][0])
        #Remove everything except year month and day
        time_output = [date.date() for date in time_output]
        self.data.index = pd.to_datetime(self.data.index.date)
        #find matching indices and get error
        matching_indices = self.data.index[self.data.index.to_series().dt.floor('D').isin(time_output)]
        matching_close_prices = self.data['Close'][matching_indices]
        mape_error = mean_absolute_percentage_error(matching_close_prices.values,
                                                    data[self.crypt_name]['price'][0:len(matching_close_prices)])
        #write data to file
        if os.path.exists("crypto_mape.yaml"):
            with open("crypto_mape.yaml", 'r') as file:
                err_data = yaml.safe_load(file)
        else:
            err_data = {}
        err_data[self.crypt_name] = [float(mape_error)]
        with open("crypto_mape.yaml", 'w') as file:
            yaml.dump(err_data, file)

        #95% 
        ci_upper, ci_lower = self.rolling_95_pct_ci(self.data['Close'])
        #Plot data
        look_back = -self.n_outputs - 10
        plt.rcParams['font.weight'] = 'bold'
        plt.figure(figsize=(14,10))
        plt.fill_between(ci_upper.index.to_numpy()[look_back:], ci_upper.to_numpy()[look_back:], 
                         ci_lower.to_numpy()[look_back:], 
                         color='lightblue', alpha=0.5, label='rolling 95% ci')
        plt.plot(time_output,data[self.crypt_name]['price'],marker='.',markersize=10,label='Predicted')
        plt.plot(self.data.index[look_back:].to_numpy(),self.data['Close'].iloc[look_back:].to_numpy(),marker='.',markersize=10,label='Actual')
        plt.title(f'{self.crypt_name} MAPE: {round(mape_error*100,3)}% | data length: {len(self.data["Close"])} samples')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(axis='y')
        plt.legend()
        plt.tight_layout()
        if not os.path.exists('figures'):
            os.mkdir('figures')
        save_path = os.path.join(os.getcwd(),'figures',f'{self.crypt_name}_predicted_vs_actual.png')
        plt.savefig(save_path,dpi=350)
        plt.close()

    def plot_error(self):
        if os.path.exists('crypto_mape.yaml'):
            with open("crypto_mape.yaml", 'r') as file:
                data = yaml.safe_load(file)

            # Filter out data points with errors above 1
            filtered_data = {crypt: error for crypt, error in data.items() if error[0] <= 1}
            # Extract crypto names and errors
            crypto_names = list(filtered_data.keys())
            errors = [filtered_data[crypt][0] for crypt in crypto_names]
            # Sort the filtered data in ascending order
            sorted_data = sorted(zip(crypto_names, errors), key=lambda x: x[1])
            # Extract sorted crypto names and errors
            crypto_names_sorted, errors_sorted = zip(*sorted_data)
            # Create a bar plot
            plt.figure(figsize=(12, 10))
            plt.bar(crypto_names_sorted, errors_sorted)
            plt.xlabel("Cryptos")
            plt.ylabel("MAPE")
            plt.title("Error by Crypto")
            plt.xticks(rotation=45, ha='right')
            # save the plot
            plt.tight_layout()
            plt.savefig('error_plot.png',dpi=400)
            plt.close()

    def correlate_len_error(self):
        #MAPE
        if os.path.exists("crypto_mape.yaml"):
            with open("crypto_mape.yaml", 'r') as file:
                err_data = yaml.safe_load(file)
        #PRED PRICE
        if os.path.exists("crypto_pre_error.yaml"):
            with open("crypto_pre_error.yaml", 'r') as file:
                price_err = yaml.safe_load(file)
        price_len = []
        error_val = []
        sample_entropy_list = []
        fractal_list = []


        #MAPE vs. ERR correlate
        for key, value in err_data.items():
            if value[0] < 1:
                print(f"Key: {key}, Value: {value}")
                if key != "SP":
                    crypt_name = key + '-USD'
                    temp = yf.Ticker(crypt_name)
                    df_crypt = temp.history(period = 'max', interval="1d")
                    #Sample entropy
                    avg_price = (df_crypt['High'] + df_crypt['Close'] + df_crypt['Low']) / 3
                    sample_entropy_list.append(sampen2(avg_price.values)[2][1])
                    #Fractal Dimension 
                    fractal_list.append(calculate_fractal_dimension(avg_price))

                    price_len.append(len(temp.history(period = 'max', interval="1d")))
                    error_val.append(value[0])
        sample_entropy_list = [value if value is not None else 0 for value in sample_entropy_list]
        data = pd.DataFrame({"MAPE":error_val,"data_len":price_len, 
                             'sampEn': sample_entropy_list, 'fractal_dim':fractal_list})
        # data = data[data['MAPE'] <= 35] #for visualization
        print('ERROR VS. MAPE DF')
        # print(data.sort_values(by=['data_len'],ascending=False))
        print(data)
        r_val, p_val = pearsonr(data['MAPE'],data['data_len'])
        r_val_se, p_val_se = pearsonr(data['MAPE'],data['sampEn'])
        r_val_frac, p_val_frac = pearsonr(data['MAPE'],data['fractal_dim'])

        #PRED PRICE
        crypt_7_day_pos_neg = {}
        correct_classify = 0
        total_classify = 0
        for key, value in price_err.items():
            if key != "SP":
                crypt_name = key + '-USD'
                temp = yf.Ticker(crypt_name)
                price = temp.history(period=f'{str(self.n_outputs)}d')
                if value['price'][0] < value['price'][-1]:
                    if (value['price'][0] < value['price'][-1]) and (price['Close'].iloc[0] < price['Close'].iloc[1]):
                        crypt_7_day_pos_neg[key] = 1
                        correct_classify += 1
                    else:
                        crypt_7_day_pos_neg[key] = 0
                    total_classify +=1 
                    print(crypt_7_day_pos_neg)
        print('========================================================')
        print(f'PROPORTION CORRECT ON THOSE THAT THE MODEL PREDICTED WOULD BE POSITIVE: {(correct_classify / total_classify)*100}%')
        #plot
        g = regplot(data,x='MAPE',y='data_len',ci=None)
        plt.text(0.9, 0.9, f'r^2 = {r_val**2:.2f}\np = {p_val:.2e}', transform=g.transAxes, ha='center')
        print(f"Correlation between MAPE vs. Length of the Price Data: {(r_val,p_val)}")
        print('========================================================')
        plt.tight_layout()
        plt.savefig('correl_mape_data_len.png',dpi=400)
        plt.close()
        #sampen vs avg price
        g = regplot(data,x='MAPE',y='sampEn',ci=None)
        plt.text(0.9, 0.9, f'r^2 = {r_val_se**2:.2f}\np = {p_val_se:.2e}', transform=g.transAxes, ha='center')
        print(f"Correlation between MAPE vs. Sample Entropy of the Price Data: {(r_val_se,p_val_se)}")
        print('========================================================')
        plt.tight_layout()
        plt.savefig('correl_mape_vs_sampEn.png',dpi=400)
        plt.close()
        #power law vs avg price
        g = regplot(data,x='MAPE',y='fractal_dim',ci=None)
        plt.text(0.9, 0.9, f'r^2 = {r_val_frac**2:.2f}\np = {p_val_frac:.2e}', transform=g.transAxes, ha='center')
        print(f"Correlation between MAPE vs. Fractal Dimension of the Price Data: {(r_val_frac,p_val_frac)}")
        print('========================================================')
        plt.tight_layout()
        plt.savefig('correl_mape_vs_frac_dim.png',dpi=400)
        plt.close()

    def plot_pct_change_all_cryptos(self):
        with open('crypto_pre_error.yaml', 'r') as file:
            data = yaml.safe_load(file)

        # Cryptocurrency keys
        available_crypt = list(data.keys())
        list_crypt = ['BTC', 'ETH', 'ADA', 'MATIC', 'DOGE', 'SOL', 'DOT', 'SHIB', 'TRX', 'FIL', 'LINK',
                    'APE', 'MANA', "AVAX", "ZEC", "ICP", "FLOW", "EGLD", "XTZ", "LTC", "XRP"]
        list_crypt = [crypt for crypt in list_crypt if crypt in available_crypt]

        # Create a dictionary to store average percentage change and standard deviation for each cryptocurrency
        dfs = []

        # Iterate through the cryptocurrency keys
        for crypt in list_crypt:
            # Check if 'price' array exists
            if 'price' in data[crypt]:
                # Filter arrays to ensure they have the same length as 'price'
                filtered_data = {key: value[:len(data[crypt]['price'])] for key, value in data[crypt].items()}

                time_entry = data[crypt]['time'][0]# datetime.strptime(data[crypt]['time'][0], '%Y-%m-%d %H:%M:%S.%f')
                time_array = [time_entry + timedelta(days=i) for i in range(len(data[crypt]['price']))]
                
                # Convert to DataFrame and set the column header to the cryptocurrency name
                df_crypt = pd.DataFrame(filtered_data['price'], columns=[crypt],index=pd.to_datetime(time_array).date)
                df_crypt[crypt] = df_crypt[crypt].pct_change()*100
                # Append DataFrame to the list
                dfs.append(df_crypt)

        # Concatenate DataFrames along the columns axis
        result_df = pd.concat(dfs, axis=1).dropna()

        row_std = result_df.std(axis=1)
        row_mean = result_df.mean(axis=1)

        # Plot the mean with standard deviation as fill between
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams['font.size'] = 14
        plt.figure(figsize=(12, 10))
        plt.plot(row_mean, label='Mean', color='blue', linewidth=3)
        plt.fill_between(result_df.index, row_mean - row_std, row_mean + row_std, alpha=0.6, color='lightblue')
        # plt.title('Mean with Standard Deviation as Fill Between')
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.ylabel('Predicted Percentage Change For All Cryptos')
        plt.hlines(xmin=result_df.index[0], xmax=result_df.index[-1],y=0, linewidth=3, color='k')
        plt.grid(True)
        plt.tight_layout()

        # Save the plot to a file
        plt.savefig('mean_with_std_plot.png')
        plt.close()

    def prediction_pos_neg(self,):
        pass

    def run_analysis(self):
        if os.path.exists('crypto_pre_error.yaml') and argv[2] == "test":
            self.check_output()
            self.plot_error()
        elif argv[2] == "correlate":
            self.plot_pct_change_all_cryptos()
            self.correlate_len_error()
        else:
            # Prepare data for training
            X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(self.data)
            self.train_model(X_train, y_train, X_val, y_val,X_test,y_test)
            curr_loss = self.evaluate_model(X_test,y_test)
            # Make prediction for tomorrow
            prediction, last_date, test = self.predict(self.data)
            print(curr_loss)
            self.plot_pct_change()
            self.save_previous_days_data()

def main():
    if argv[1] == 'all':
        list_crypt = ['BTC','ETH','ADA','MATIC','DOGE',
                    'SOL','DOT','SHIB','TRX','FIL','LINK',
                    'APE','MANA',"AVAX","ZEC","ICP","FLOW",
                    "EGLD","XTZ","LTC","XRP"] # "SP",
        for name in tqdm(sorted(list_crypt)):
            try:
                changePricePredictor(crypt=name,
                                    n_features=10, 
                                    n_steps=128, 
                                    n_outputs=21, 
                                    n_epochs=500, 
                                    batch_size=256).run_analysis()
                if argv[2] == "correlate":
                    break
            except Exception as e:
                print(Fore.RED,Style.BRIGHT,f'{name} did not complete: {e}',Style.RESET_ALL)
    else:
        changePricePredictor(crypt=argv[1],
                            n_features=10, 
                            n_steps=128, 
                            n_outputs=21, 
                            n_epochs=500, 
                            batch_size=256).run_analysis()
if __name__ == "__main__":
    main()