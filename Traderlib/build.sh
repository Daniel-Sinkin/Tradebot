c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) ema_module.cpp -o ema_module$(python3-config --extension-suffix) \
-L$(python3-config --prefix)/lib -I$(python3-config --prefix)/include/python3.12 -lpython3.12
