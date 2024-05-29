#!/bin/bash
python crypto_deep_many_features.py all test && \
python crypto_deep_many_features.py all correlate && \
python change_price_over_time.py --name all --extension True
