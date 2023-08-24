import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import ta
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
from datetime import datetime
# from keras_tuner.engine.hyperparameters import HyperParameters
import yaml
def create_lstm_model(hp, n_steps, n_features, n_outputs):
    activation_choice = hp.Choice('activation', values=['relu', 'leaky_relu', 'tanh'])
    model = tf.keras.models.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hp.Int('units', min_value=50, max_value=200, step=50),
                                                           activation=activation_choice, return_sequences=False,
                                                           input_shape=(n_steps, n_features))),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(n_outputs, activation="linear")
    ])

    optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop'])
    if optimizer_choice == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Float('adam_learning_rate', min_value=0.0001, max_value=0.001, sampling='log'))
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp.Float('rmsprop_learning_rate', min_value=0.0001, max_value=0.001, sampling='log'))
    
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mean_squared_error']) #tf.keras.metrics.RootMeanSquaredError()
    
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
    crypto_data[crypto_name] = {'price': [float(price)], 'error': [float(error)], "time": [datetime.now()]}
    with open(filename, 'w') as file:
        yaml.dump(crypto_data, file)

def find_crypto_data(filename, crypto_name):
    crypto_data = load_data_from_yaml(filename)
    return crypto_data.get(crypto_name)
    
class changePricePredictor:
    def __init__(self, crypt, n_features, n_steps, n_outputs, n_epochs, batch_size):
        self.crypt_name = crypt
        # self.n_features = n_features
        self.n_steps = n_steps
        self.n_outputs = n_outputs
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        crypt_name = crypt + '-USD'
        temp = yf.Ticker(crypt_name)
        price_data = temp.history(period = 'max', interval="1d")
        print(Fore.GREEN,f'NUMBER OF SAMPLES FOR {crypt_name}: {len(price_data)}',Style.RESET_ALL)
        self.features = ['Close','Open', 'High', 'Low','Volume', 'Dividends', 'Stock Splits',
                                    'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em',
                                    'volume_sma_em', 'volume_vpt', 'volume_vwap', 'volume_mfi',
                                    'volume_nvi', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
                                    'volatility_bbw', 'volatility_bbp', 'volatility_bbhi',
                                    'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl',
                                    'volatility_kcw', 'volatility_kcp', 'volatility_kchi',
                                    'volatility_kcli', 'volatility_dcl', 'volatility_dch', 'volatility_dcm',
                                    'volatility_dcw', 'volatility_dcp', 'volatility_atr', 'volatility_ui',
                                    'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
                                    'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
                                    'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_ind_diff',
                                    'trend_trix', 'trend_mass_index', 'trend_dpo', 'trend_kst',
                                    'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
                                    'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',
                                    'trend_stc', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci',
                                    'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',
                                    'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up',
                                    'trend_psar_down', 'trend_psar_up_indicator',
                                    'trend_psar_down_indicator', 'momentum_rsi', 'momentum_stoch_rsi',
                                    'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'momentum_tsi',
                                    'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',
                                    'momentum_ao', 'momentum_roc', 'momentum_ppo', 'momentum_ppo_signal',
                                    'momentum_ppo_hist', 'momentum_pvo', 'momentum_pvo_signal',
                                    'momentum_pvo_hist', 'momentum_kama', 'others_dr', 'others_dlr',
                                    'others_cr']
        self.non_close_features = ['Volume',
                                    'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em',
                                    'volume_sma_em', 'volume_vpt', 'volume_vwap', 'volume_mfi',
                                    'volume_nvi', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
                                    'volatility_bbw', 'volatility_bbp', 'volatility_bbhi',
                                    'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl',
                                    'volatility_kcw', 'volatility_kcp', 'volatility_kchi',
                                    'volatility_kcli', 'volatility_dcl', 'volatility_dch', 'volatility_dcm',
                                    'volatility_dcw', 'volatility_dcp', 'volatility_atr', 'volatility_ui',
                                    'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
                                    'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
                                    'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_ind_diff',
                                    'trend_trix', 'trend_mass_index', 'trend_dpo', 'trend_kst',
                                    'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
                                    'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',
                                    'trend_stc', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci',
                                    'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',
                                    'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up',
                                    'trend_psar_down', 'trend_psar_up_indicator',
                                    'trend_psar_down_indicator', 'momentum_rsi', 'momentum_stoch_rsi',
                                    'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'momentum_tsi',
                                    'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',
                                    'momentum_ao', 'momentum_roc', 'momentum_ppo', 'momentum_ppo_signal',
                                    'momentum_ppo_hist', 'momentum_pvo', 'momentum_pvo_signal',
                                    'momentum_pvo_hist', 'momentum_kama', 'others_dr', 'others_dlr',
                                    'others_cr'] #'Open', 'High', 'Low', 'Dividends', 'Stock Splits',
        self.n_features = len(self.non_close_features)
        self.data = ta.add_all_ta_features(
            price_data,
            open="Open",
            high="High",
            close='Close',
            low='Low',
            volume='Volume',
            fillna=True
        )
        print(self.data.columns)

    def prepare_data(self, data):
        # Extract relevant features
        data = self.data[self.features]

        # Scale data
        # self.scaler2 = MinMaxScaler(feature_range=(0, 1))
        self.scaler2 = StandardScaler()
        # self.scaler1 = MinMaxScaler(feature_range=(0, 1))
        # self.scaler1 = StandardScaler()
        
        #Close price
        data_close = data['Close'].pct_change().fillna(0).to_numpy().reshape(-1, 1) #close price
        # data_close = self.scaler1.fit_transform(data_close)

        data_non_close = data[self.non_close_features]
        data_non_close = self.scaler2.fit_transform(data_non_close)
        data = np.concatenate((data_close, data_non_close), axis=1)

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

    def train_model(self, X_train, y_train, X_val, y_val):
        if os.path.exists(f"{self.crypt_name}_lstm_model.h5"):
            self.model = load_model(f"{self.crypt_name}_lstm_model.h5")
        else:
            # model = self.create_model()
            #TUNE LSTM
            tuner = RandomSearch(
                lambda hp: create_lstm_model(hp, self.n_steps, self.n_features, self.n_outputs),
                objective='val_loss',
                max_trials=20,
                directory=f'{self.crypt_name}_lstm_hp',
                project_name='lstm_hyperparameter_tuning',
                # overwrite=True
            )
            tuner.search(x=X_train, y=y_train,
                    epochs=75,
                    validation_data=(X_val, y_val))
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            self.best_model = tuner.get_best_models(num_models=1)[0]
            print(f"Best Hyperparameters: {best_hps.values}")
            print(f'best model error: {self.best_model.evaluate(X_val, y_val)}')
            self.model = self.best_model
            self.best_model.save(f"{self.crypt_name}_lstm_model.h5")

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
        data = data[self.features]
        # data_close = data['Close'].pct_change().fillna(method='bfill').to_numpy().reshape(-1, 1) #pct_change
        data_close = data['Close'].pct_change().to_numpy().reshape(-1, 1) #close
        # data_close = self.scaler1.transform(data_close)
        data_non_close = data[self.non_close_features]
        data_non_close = self.scaler2.transform(data_non_close)
        data = np.concatenate((data_close, data_non_close), axis=1)

        #Predict the future after test data
        if argv[2] == "test":
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
    
    def plot_pct_change(self):
        last_week = self.data['Close'].iloc[-14:]
        tomorrow = last_week[-1] + (last_week[-1] * self.y_pred)
        self.data.index = pd.to_datetime(self.data.index)
        # Calculate the next day's datetime
        next_day_datetime = self.data.index[-1] + pd.Timedelta(days=1)
        print(next_day_datetime)
        last_week[next_day_datetime] = tomorrow
        plt.figure()
        plt.plot(last_week.index[:-1], last_week[:-1], color='blue', marker='o', label='Price')
        plt.plot(last_week.index[-1], last_week[-1], color='green', marker='o', label='Predicted Price')
        plt.plot(last_week.index[-2:], last_week[-2:], color='blue')  # Connect the last two points with a blue line
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{self.crypt_name} price for {datetime.now() + timedelta(days=1)} is {tomorrow[0]}')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        if not os.path.exists('figures'):
            os.mkdir('figures')
        save_path = os.path.join(os.getcwd(),'figures',f'{self.crypt_name}_tomorrow_price.png')
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
        tomorrow = self.data['Close'].iloc[-1:].values + (self.data['Close'].iloc[-1:].values * self.y_pred)
        update_crypto_data(filename, self.crypt_name, tomorrow[0], self.loss_output)
        if os.path.exists('predictions'):
            os.mkdir('predictions')
        # np.savetxt(os.path.join(os.getcwd(),'predictions',f'{self.crypt_name}_prediction.txt'), self.y_pred[0][0], fmt='%.6f')

    def run_analysis(self):
        # Prepare data for training
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(self.data)
        self.train_model(X_train, y_train, X_val, y_val)
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
                    "EGLD","XTZ","LTC"]
        for name in tqdm(sorted(list_crypt)):
            try:
                changePricePredictor(crypt=name,
                                    n_features=10, 
                                    n_steps=60, 
                                    n_outputs=1, 
                                    n_epochs=500, 
                                    batch_size=128).run_analysis()
            except:
                print(f'too many nans for {name}')
    else:
        changePricePredictor(crypt=argv[1],
                            n_features=10, 
                            n_steps=60, 
                            n_outputs=1, 
                            n_epochs=500, 
                            batch_size=128).run_analysis()
if __name__ == "__main__":
    main()