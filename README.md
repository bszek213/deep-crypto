# Cryptocurrency forecasting with LSTM 

Deep learning to forecast cryptocurrency prices 7 days in the future. Here are all the cryptos that have models:
'BTC','ETH','ADA','MATIC','DOGE','SOL','DOT','SHIB','TRX','FIL','LINK','APE','MANA',"AVAX","ZEC","ICP","FLOW","EGLD","XTZ","LTC"

Currently, there are 182 features to help with the forecasting of the close price
## Usage

```bash
python3 crypto_deep_many_features.py all test #check the output for all cryptos
python3 crypto_deep_many_features.py all notest #perform training or create future prediction
python3 crypto_deep_many_features.py BTC test #check output for individiual crypto, in this case BTC.
python3 crypto_deep_many_features.py BTC notest #perform training or create future forecast for individual crypto, in this case BTC.
#change price over time
python3 change_price_over_time.py --name all_cryptos --extension True #run all cryptos from a predefined list
python3 change_price_over_time.py --name trending --extension False #run analysis on trending stocks from Yahoo Finance
python3 change_price_over_time.py --name BTC --extension True #run individual crypto
python3 change_price_over_time.py --name ^DJI --extension False #run individual stock
```
### Correlated Features
![](https://github.com/bszek213/deep-crypto/blob/dev/correlation_heatmap.png)
### Forecasting error
![](https://github.com/bszek213/deep-crypto/blob/dev/error_plot.png)
### Forecasting examples of BTC
![](https://github.com/bszek213/deep-crypto/blob/dev/figures/BTC_future_price.png)
### Relationship between Data Length and MAPE
![](https://github.com/bszek213/deep-crypto/blob/dev/correl_mape_data_len.png)
### Month Analysis
Month
![](https://github.com/bszek213/deep-crypto/blob/dev/price_change/BTC_change_month.png)
<!-- Week
![](https://github.com/bszek213/deep-crypto/blob/dev/price_change/BTC_change_week.png) -->
### Rainbow Log plot of BTC
![](https://github.com/bszek213/deep-crypto/blob/dev/figures_rainbow/BTC_rainbow.png)
### Shannon Entropy on the log returns on cryptos and stock market
![](https://github.com/bszek213/deep-crypto/blob/dev/Entropy_cryptos_stock_market.png)
### Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.