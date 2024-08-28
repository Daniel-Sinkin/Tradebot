import timeit

import numpy as np

import TraderBot.ema_module as ema
from TraderBot.traderlib import ema_filter

# Example data - significantly larger list
data = np.random.rand(1000000).astype(np.float32)  # 1 million random floats
span = 50

# Profiling the ema_filter function
python_time = timeit.timeit(lambda: ema_filter(data.tolist(), span), number=100)
print(f"Python EMA Filter Time: {python_time:.6f} seconds")

# Profiling the C++ module function with NumPy array
cpp_time = timeit.timeit(lambda: ema.computeEMA(data, span), number=100)
print(f"C++ EMA Module Time: {cpp_time:.6f} seconds")
