#!/bin/bash

# Avoids linker errors
export SDKROOT=$(xcrun --show-sdk-path)

# Gets absolute path
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Moves the output one up and into the traderbot folder
OUTPUT_DIR="${SCRIPT_DIR}/../traderbot"

SOURCE_FILE="${SCRIPT_DIR}/traderlib.cpp"

# Update the module name in the output file
c++ -O3 -Wall -shared -std=c++11 -fPIC \
$(python3 -m pybind11 --includes) \
-I${SDKROOT}/usr/include \
-L${SDKROOT}/usr/lib \
-L$(python3-config --prefix)/lib \
-lpython3.12 \
"${SOURCE_FILE}" -o "${OUTPUT_DIR}/ta_module$(python3-config --extension-suffix)"
