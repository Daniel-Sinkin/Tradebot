#!/bin/bash

# Set the SDK path
export SDKROOT=$(xcrun --show-sdk-path)

# Get the absolute path to the Traderlib directory
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Define the output directory as ../TraderBot relative to the Traderlib directory
OUTPUT_DIR="${SCRIPT_DIR}/../TraderBot"

# Define the input source file path (ema_module.cpp) within the Traderlib directory
SOURCE_FILE="${SCRIPT_DIR}/ema_module.cpp"

# Compile the C++ module and output to the TraderBot directory
c++ -O3 -Wall -shared -std=c++11 -fPIC \
$(python3 -m pybind11 --includes) \
-I${SDKROOT}/usr/include \
-L${SDKROOT}/usr/lib \
-L$(python3-config --prefix)/lib \
-lpython3.12 \
"${SOURCE_FILE}" -o "${OUTPUT_DIR}/ema_module$(python3-config --extension-suffix)"