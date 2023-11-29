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
import yaml
from tensorflow.keras.callbacks import EarlyStopping
from seaborn import regplot, heatmap
from scipy.stats import pearsonr

def create_lstm_model(hp, n_steps, n_features, n_outputs):
    activation_choice = hp.Choice('activation', values=['relu', 'leaky_relu', 'tanh'])
    regularizer_strength_l1 = hp.Float('regularizer_strength_l1', min_value=1e-6, max_value=1e-2, sampling='log')
    regularizer_strength_l2 = hp.Float('regularizer_strength_l2', min_value=1e-6, max_value=1e-2, sampling='log')
    
    regularizer_l1 = tf.keras.regularizers.l1(regularizer_strength_l1)
    regularizer_l2 = tf.keras.regularizers.l2(regularizer_strength_l2)
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hp.Int('units', min_value=1, max_value=20, step=1),
                                                           activation=activation_choice, return_sequences=False,
                                                           kernel_regularizer=regularizer_l2, recurrent_regularizer=regularizer_l1,
                                                           input_shape=(n_steps, n_features))),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(n_outputs, 
                              hp.Choice('dense_activation', values=['linear','leaky_relu', 'tanh']))
    ])

    optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
    if optimizer_choice == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Float('adam_learning_rate', min_value=0.0001, max_value=0.001, sampling='log'))
    elif optimizer_choice == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp.Float('rmsprop_learning_rate', min_value=0.0001, max_value=0.001, sampling='log'))
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=hp.Float('sgd_learning_rate', min_value=0.0001, max_value=0.001, sampling='log'))
    
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

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
        if len(temp.history(period = 'max', interval="1d")) < 1:
            print(f'{crypt_name} has no data. No connection to yfinance')
            exit()
        price_data = temp.history(period = 'max', interval="1d")
        #change the date out of utc to mine
        current_date = datetime.now().date()
        new_index = pd.date_range(end=current_date, periods=len(price_data.index), freq='D')
        price_data.index = new_index
        
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
        self.non_close_features = ['Volume','Open', 'High', 'Low',
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
        # print(self.data.columns)

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
        # plt.hist(data_close,bins=400)
        # plt.show()
        # data_close = self.scaler1.fit_transform(data_close)

        data_non_close = data[self.non_close_features]
        
        #Remove correlated features
        threshold = 0.95
        data_non_close.corr()
        correlation_matrix = data_non_close.corr()
        mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        to_drop = [column for column in correlation_matrix.columns if any(correlation_matrix.loc[column, mask[:, correlation_matrix.columns.get_loc(column)]] > threshold)]
        print(f'Features to be removed: {to_drop}')
        #features to keep
        self.non_close_features = [item for item in self.non_close_features if item not in to_drop]
        # Remove highly correlated features
        data_non_close = data_non_close.drop(columns=to_drop)
        # Create and save a heatmap plot
        plt.figure(figsize=(10, 10))
        heatmap(correlation_matrix, cmap="coolwarm", mask=mask)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png")
        plt.close()

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
        save_path = os.path.join(os.getcwd(),'model_loc')
        if os.path.isdir(save_path):
            print('path exists')
        else:
            os.mkdir('model_loc')
        save_path = os.path.join(save_path,f"{self.crypt_name}_lstm_model.h5")
        if os.path.exists(save_path):
            self.model = load_model(save_path)
        else:
            # model = self.create_model()
            #TUNE LSTM
            tuner = RandomSearch(
                lambda hp: create_lstm_model(hp, self.n_steps, self.n_features, self.n_outputs),
                objective='val_loss',
                max_trials=30,
                directory=f'{self.crypt_name}_lstm_hp',
                project_name='lstm_hyperparameter_tuning',
                # overwrite=True
            )
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            tuner.search(x=X_train, y=y_train,
                    epochs=100,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping])
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            self.best_model = tuner.get_best_models(num_models=1)[0]
            #fit tuned model
            self.best_model.fit(X_train, y_train, epochs=75, validation_data=(X_val, y_val),
            callbacks=[early_stopping])
            #write best hyperparameters to file
            file_path = 'best_hp.txt'
            content_to_append = f"{self.crypt_name} Best Hyperparameters: {best_hps.values}"
            with open(file_path, 'a') as file:
                file.write(content_to_append + '\n')
            #save model
            self.model = self.best_model
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
        data = data[self.features]
        # data_close = data['Close'].pct_change().fillna(method='bfill').to_numpy().reshape(-1, 1) #pct_change
        data_close = data['Close'].pct_change().to_numpy().reshape(-1, 1) #close
        # data_close = self.scaler1.transform(data_close)
        data_non_close = data[self.non_close_features]
        data_non_close = self.scaler2.transform(data_non_close)
        data = np.concatenate((data_close, data_non_close), axis=1)

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
    
    def plot_pct_change(self):
        last_week = self.data['Close'].iloc[-14:]
        price_val = last_week[-1]
        save_price = [] 
        for val in self.y_pred:
            tomorrow = price_val + (price_val * val)
            price_val = tomorrow
            save_price.append(tomorrow)

        self.data.index = pd.to_datetime(self.data.index)
        # Calculate the next days datetime
        start_datetime = self.data.index[-1] + pd.Timedelta(days=1)
        next_days = [start_datetime + pd.Timedelta(days=i) for i in range(len(self.y_pred))]

        plt.figure()
        plt.plot(last_week.index[:-1].to_numpy(), last_week[:-1].to_numpy(), color='blue', marker='o', label='Price')
        plt.plot(next_days, save_price, color='green', marker='o', label='Predicted Price')
        plt.plot(last_week.index[-2:].to_numpy(), last_week[-2:].to_numpy(), color='blue')  # Connect the last two points with a blue line
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{self.crypt_name} price for the next {len((self.y_pred))} days')
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
        mape_error = mean_absolute_percentage_error(matching_close_prices.values,data[self.crypt_name]['price'][0:len(matching_close_prices)])
        #write data to file
        if os.path.exists("crypto_mape.yaml"):
            with open("crypto_mape.yaml", 'r') as file:
                err_data = yaml.safe_load(file)
        else:
            err_data = {}
        err_data[self.crypt_name] = [float(mape_error)]
        with open("crypto_mape.yaml", 'w') as file:
            yaml.dump(err_data, file)

        #Plot data
        plt.plot(time_output,data[self.crypt_name]['price'],marker='.',markersize=10,label='Predicted')
        plt.plot(self.data.index[-21:].to_numpy(),self.data['Close'].iloc[-21:].to_numpy(),marker='.',markersize=10,label='Actual')
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
            plt.figure(figsize=(10, 6))
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
        #MAPE vs. ERR correlate
        for key, value in err_data.items():
            if value[0] < 100:
                print(f"Key: {key}, Value: {value}")
                crypt_name = key + '-USD'
                temp = yf.Ticker(crypt_name)
                price_len.append(len(temp.history(period = 'max', interval="1d")))
                error_val.append(value[0])
        data = pd.DataFrame({"MAPE":error_val,"data_len":price_len})
        data.set_index(pd.Index([key for key, value in err_data.items() if value[0] < 100]), inplace=True)
        print('ERROR VS. MAPE DF')
        print(data.sort_values(by=['data_len'],ascending=False))
        r_val, p_val = pearsonr(data['MAPE'],data['data_len'])

        #PRED PRICE
        crypt_7_day_pos_neg = {}
        correct_classify = 0
        total_classify = 0
        for key, value in price_err.items():
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
        g = regplot(data,x='MAPE',y='data_len')
        plt.text(0.9, 0.9, f'r^2 = {r_val**2:.2f}\np = {p_val:.2e}', transform=g.transAxes, ha='center')
        print(f"Correlation between MAPE vs. Length of the Price Data: {(r_val,p_val)}")
        print('========================================================')
        plt.tight_layout()
        plt.savefig('correl_mape_data_len.png',dpi=400)
        plt.close()

    def prediction_pos_neg(self,):
        pass

    def run_analysis(self):
        if os.path.exists('crypto_pre_error.yaml') and argv[2] == "test":
            self.check_output()
            self.plot_error()
        elif argv[2] == "correlate":
            self.correlate_len_error()
        else:
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
                                    n_steps=128, 
                                    n_outputs=7, 
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
                            n_outputs=7, 
                            n_epochs=500, 
                            batch_size=256).run_analysis()
if __name__ == "__main__":
    main()