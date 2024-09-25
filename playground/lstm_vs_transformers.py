import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import tensorflow as tf
# def load_data():
#     data = yf.download('BTC-USD')
#     data = data[['Close']]  # We'll use the 'Close' price for prediction
#     data.to_csv('data.csv',index=False)

# load_data()

# Parameters for data processing
sequence_length = 60  # Use past 60 days to predict the future price
forecast_horizon = 7  # Predict 7 days into the future

def preprocess_data(data, sequence_length, forecast_horizon):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X = []
    y = []
    for i in range(sequence_length, len(scaled_data) - forecast_horizon + 1):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i:i+forecast_horizon])

    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

X, y, scaler = preprocess_data(data, sequence_length, forecast_horizon)

# Split into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f'Training samples: {X_train.shape[0]}')
print(f'Testing samples: {X_test.shape[0]}')

def build_lstm_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1], X_train.shape[2])))
    
    # Tune number of LSTM layers and units
    for i in range(hp.Int('lstm_layers', 1, 3)):
        model.add(layers.LSTM(
            units=hp.Int(f'lstm_units_{i}', min_value=32, max_value=256, step=32),
            return_sequences=True if i < hp.Int('lstm_layers', 1, 3) - 1 else False
        ))
        model.add(layers.Dropout(rate=hp.Float(f'dropout_{i}', 0.1, 0.5, step=0.1)))
    
    model.add(layers.Dense(units=forecast_horizon))
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        ),
        loss='mse',
        metrics=['mae']
    )
    return model

def build_transformer_model(hp):
    input_layer = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
    
    # Positional Encoding
    position_embedding = layers.Embedding(
        input_dim=X_train.shape[1],
        output_dim=hp.Int('embed_dim', 32, 128, step=32)
    )(tf.range(start=0, limit=X_train.shape[1], delta=1))
    x = layers.Add()([input_layer, position_embedding])
    
    # Transformer Blocks
    for i in range(hp.Int('transformer_blocks', 1, 3)):
        attention_output = layers.MultiHeadAttention(
            num_heads=hp.Int('num_heads', 2, 8, step=2),
            key_dim=hp.Int('key_dim', 32, 128, step=32)
        )(x, x)
        attention_output = layers.Dropout(rate=hp.Float(f'dropout_{i}', 0.1, 0.5, step=0.1))(attention_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        ffn = keras.Sequential([
            layers.Dense(units=hp.Int('ffn_units', 32, 256, step=32), activation='relu'),
            layers.Dense(units=hp.Int('embed_dim', 32, 128, step=32))
        ])
        ffn_output = ffn(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    x = layers.GlobalAveragePooling1D()(x)
    output_layer = layers.Dense(units=forecast_horizon)(x)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        ),
        loss='mse',
        metrics=['mae']
    )
    return model

# Define tuner parameters
tuner_lstm = kt.RandomSearch(
    build_lstm_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=1,
    directory='keras_tuner',
    project_name='bitcoin_lstm'
)

tuner_transformer = kt.RandomSearch(
    build_transformer_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=1,
    directory='keras_tuner',
    project_name='bitcoin_transformer'
)

# Early stopping to prevent overfitting
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Search for the best hyperparameters for LSTM
print("Tuning LSTM Model...")
tuner_lstm.search(
    X_train, y_train,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stop]
)

# Get the best LSTM model
best_lstm_model = tuner_lstm.get_best_models(1)[0]

# Search for the best hyperparameters for Transformer
print("Tuning Transformer Model...")
tuner_transformer.search(
    X_train, y_train,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stop]
)

# Get the best Transformer model
best_transformer_model = tuner_transformer.get_best_models(1)[0]

# Train the best LSTM model
history_lstm = best_lstm_model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Evaluate LSTM model
lstm_loss, lstm_mae = best_lstm_model.evaluate(X_test, y_test)
print(f'LSTM Test Loss: {lstm_loss}, Test MAE: {lstm_mae}')

# Train the best Transformer model
history_transformer = best_transformer_model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Evaluate Transformer model
transformer_loss, transformer_mae = best_transformer_model.evaluate(X_test, y_test)
print(f'Transformer Test Loss: {transformer_loss}, Test MAE: {transformer_mae}')


# Make predictions with LSTM
lstm_predictions = best_lstm_model.predict(X_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions.reshape(-1, 1)).reshape(-1, forecast_horizon)

# Make predictions with Transformer
transformer_predictions = best_transformer_model.predict(X_test)
transformer_predictions = scaler.inverse_transform(transformer_predictions.reshape(-1, 1)).reshape(-1, forecast_horizon)

# Prepare true values
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1, forecast_horizon)

# Plotting predictions vs true values for the first test sample
plt.figure(figsize=(12, 6))
plt.plot(range(forecast_horizon), y_test_inverse[0], label='True Prices')
plt.plot(range(forecast_horizon), lstm_predictions[0], label='LSTM Predictions')
plt.plot(range(forecast_horizon), transformer_predictions[0], label='Transformer Predictions')
plt.title('Bitcoin Price Prediction for Next 7 Days')
plt.xlabel('Day')
plt.ylabel('Price (USD)')
plt.legend()
plt.savefig('lstm_vs_transformer.png',dpi=350)

