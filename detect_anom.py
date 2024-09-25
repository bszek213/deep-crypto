import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import yfinance as yf
import matplotlib.pyplot as plt

# Load your BTC price data
# Assuming you have a CSV file with a 'price' column
temp = yf.Ticker('BTC-USD')
df = temp.history(period = 'max', interval="1d")
print(df)
# Feature scaling for unsupervised learning
scaler = StandardScaler()
df['scaled_price'] = scaler.fit_transform(df[['Close']].pct_change().fillna(0))

# Splitting the data (90% training, 10% testing)
train_size = int(0.9 * len(df))
train_data = df['scaled_price'].iloc[:train_size].values.reshape(-1, 1)
test_data = df['scaled_price'].iloc[train_size:].values.reshape(-1, 1)

# Initialize and fit Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)  # 'contamination' is the expected anomaly percentage
iso_forest.fit(train_data)

# Predict anomalies (-1 means anomaly, 1 means normal)
train_pred = iso_forest.predict(train_data)
test_pred = iso_forest.predict(test_data)

# Convert predictions to labels (1 = normal, -1 = anomaly)
train_anomalies = np.where(train_pred == -1, 1, 0)
test_anomalies = np.where(test_pred == -1, 1, 0)

# Optional: evaluate on test data
# If you have true labels for anomalies in the test set, you can use classification_report
# true_labels = [0, 0, 1, ..., 0]  # Replace with actual test labels if available
# print(classification_report(true_labels, test_anomalies))

# Output test predictions (0 = normal, 1 = anomaly)
df_test = df.iloc[train_size:].copy()
df_test['anomaly'] = test_anomalies
print(df_test[['Close', 'anomaly']])

plt.figure(figsize=(10, 6))
# Non-anomalies (anomaly == 0) - plot in black
plt.plot(df_test.index[df_test['anomaly'] == 0], df_test['Close'][df_test['anomaly'] == 0], 'k-', label='Normal')
# Anomalies (anomaly == 1) - plot in red
plt.plot(df_test.index[df_test['anomaly'] == 1], df_test['Close'][df_test['anomaly'] == 1], 'ro', label='Anomaly')
# Formatting the plot
plt.title('BTC Close Price with Anomalies')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()