#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <stdexcept>

// Function to compute EMA
pybind11::array_t<float> computeEMA(const pybind11::array_t<float> &data_array, int lookback)
{
    if (lookback <= 0)
    {
        throw std::invalid_argument("Lookback must be a positive integer.");
    }

    pybind11::buffer_info buf = data_array.request();

    if (buf.ndim != 1)
    {
        throw std::runtime_error("Input data should be a 1D array.");
    }

    const float *data_ptr = static_cast<const float *>(buf.ptr);
    size_t size = buf.size;

    if (size == 0)
    {
        return pybind11::array_t<float>(size); // Return an empty numpy array
    }

    // Create a new numpy array to hold the result
    pybind11::array_t<float> ema(size);
    auto ema_buf = ema.request();
    float *ema_ptr = static_cast<float *>(ema_buf.ptr);

    float alpha = 2.0f / (lookback + 1);
    float one_minus_alpha = 1.0f - alpha;

    ema_ptr[0] = data_ptr[0];

    for (size_t i = 1; i < size; ++i)
    {
        ema_ptr[i] = alpha * data_ptr[i] + one_minus_alpha * ema_ptr[i - 1];
    }

    return ema;
}

// Define the module
PYBIND11_MODULE(ema_module, m)
{
    m.doc() = "Module for computing Exponential Moving Average (EMA)";
    m.def("computeEMA", &computeEMA, "Compute Exponential Moving Average (EMA)",
          pybind11::arg("data_array"), pybind11::arg("lookback"));
}