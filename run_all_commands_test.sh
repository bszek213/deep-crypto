#!/bin/bash
# python crypto_deep_many_features.py all test && \
# python crypto_deep_many_features.py all correlate && \
# python change_price_over_time.py --name all --extension True

SUBFOLDER="model_loc"

# Function to check RAM usage and kill the Python process if it exceeds 80%
check_ram_usage() {
    used_mem=$(free | awk '/^Mem/ {printf("%.0f", $3/$2 * 100.0)}')
    
    if [ "$used_mem" -gt 80 ]; then
        echo "RAM usage is above 80% ($used_mem%). Python terminated."
        pkill -f "python crypto_deep_many_features.py all train"
        return 1 #terminate
    fi

    return 0  #no termination
}

while [ "$(ls -1q "$SUBFOLDER" | wc -l)" -lt 150 ]; do
    python crypto_deep_many_features.py all train &
    while ps -C python > /dev/null; do
        check_ram_usage
        if [ $? -eq 1 ]; then
            break
        fi
        sleep 5
    done
done
