# Cryptocurrency forecasting with LSTM 

Deep learning to forecast cryptocurrency prices 7 days in the future. Here are all the cryptos that have models:
'BTC','ETH','ADA','MATIC','DOGE','SOL','DOT','SHIB','TRX','FIL','LINK','APE','MANA',"AVAX","ZEC","ICP","FLOW","EGLD","XTZ","LTC"

## Usage

```bash
python3 crypto_deep_many_features.py all test #check the output for all cryptos
python3 crypto_deep_many_features.py all notest #perform training or create future prediction
python3 crypto_deep_many_features.py BTC test #check output for individiual crypto, in this case BTC.
python3 crypto_deep_many_features.py BTC notest #perform training or create future forecast for individual crypto, in this case BTC.
```
### Correlated Features
![](https://github.com/bszek213/deep-crypto/blob/dev/correlation_heatmap.png)
### Forecasting error
![](https://github.com/bszek213/deep-crypto/blob/dev/error_plot.png)
### Relationship between Data Length and MAPE
![](https://github.com/bszek213/deep-crypto/blob/dev/correl_mape_data_len.png)
### By Week and Month Analysis
Month
![](https://github.com/bszek213/deep-crypto/blob/dev/price_change/BTC_change_month.png)
Week
![](https://github.com/bszek213/deep-crypto/blob/dev/price_change/BTC_change_week.png)
### Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.