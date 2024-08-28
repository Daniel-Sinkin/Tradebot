#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <stdexcept>
#include <algorithm> // For std::max and std::min

// Function to compute Stochastic Oscillator
pybind11::array_t<float> computeStochasticOscillator(const pybind11::array_t<float> &close_prices,
                                                     const pybind11::array_t<float> &low_prices,
                                                     const pybind11::array_t<float> &high_prices,
                                                     int lookback)
{
    if (lookback <= 0)
    {
        throw std::invalid_argument("Lookback must be a positive integer.");
    }

    pybind11::buffer_info close_buf = close_prices.request();
    pybind11::buffer_info low_buf = low_prices.request();
    pybind11::buffer_info high_buf = high_prices.request();

    if (close_buf.ndim != 1 || low_buf.ndim != 1 || high_buf.ndim != 1)
    {
        throw std::runtime_error("Input data should be 1D arrays.");
    }

    if (close_buf.size != low_buf.size || close_buf.size != high_buf.size)
    {
        throw std::runtime_error("All input arrays must have the same size.");
    }

    const float *close_ptr = static_cast<const float *>(close_buf.ptr);
    const float *low_ptr = static_cast<const float *>(low_buf.ptr);
    const float *high_ptr = static_cast<const float *>(high_buf.ptr);
    size_t size = close_buf.size;

    pybind11::array_t<float> stochastic(size);
    auto stochastic_buf = stochastic.request();
    float *stochastic_ptr = static_cast<float *>(stochastic_buf.ptr);

    for (size_t i = 0; i < size; ++i)
    {
        if (i < lookback - 1)
        {
            stochastic_ptr[i] = 0.0f; // Not enough data points, return 0 or handle as needed
        }
        else
        {
            float highest_high = high_ptr[i];
            float lowest_low = low_ptr[i];
            for (int j = 0; j < lookback; ++j)
            {
                highest_high = std::max(highest_high, high_ptr[i - j]);
                lowest_low = std::min(lowest_low, low_ptr[i - j]);
            }
            stochastic_ptr[i] = 100.0f * (close_ptr[i] - lowest_low) / (highest_high - lowest_low);
        }
    }

    return stochastic;
}

// Function to compute Resistance Level
pybind11::array_t<float> computeResistance(const pybind11::array_t<float> &high_prices, int lookback)
{
    if (lookback <= 0)
    {
        throw std::invalid_argument("Lookback must be a positive integer.");
    }

    pybind11::buffer_info buf = high_prices.request();

    if (buf.ndim != 1)
    {
        throw std::runtime_error("Input data should be a 1D array.");
    }

    const float *high_ptr = static_cast<const float *>(buf.ptr);
    size_t size = buf.size;

    pybind11::array_t<float> resistance(size);
    auto resistance_buf = resistance.request();
    float *resistance_ptr = static_cast<float *>(resistance_buf.ptr);

    for (size_t i = 0; i < size; ++i)
    {
        if (i < lookback - 1)
        {
            resistance_ptr[i] = 0.0f; // Not enough data points, return 0 or handle as needed
        }
        else
        {
            float highest_high = high_ptr[i];
            for (int j = 0; j < lookback; ++j)
            {
                highest_high = std::max(highest_high, high_ptr[i - j]);
            }
            resistance_ptr[i] = highest_high;
        }
    }

    return resistance;
}

// Function to compute Support Level
pybind11::array_t<float> computeSupport(const pybind11::array_t<float> &low_prices, int lookback)
{
    if (lookback <= 0)
    {
        throw std::invalid_argument("Lookback must be a positive integer.");
    }

    pybind11::buffer_info buf = low_prices.request();

    if (buf.ndim != 1)
    {
        throw std::runtime_error("Input data should be a 1D array.");
    }

    const float *low_ptr = static_cast<const float *>(buf.ptr);
    size_t size = buf.size;

    pybind11::array_t<float> support(size);
    auto support_buf = support.request();
    float *support_ptr = static_cast<float *>(support_buf.ptr);

    for (size_t i = 0; i < size; ++i)
    {
        if (i < lookback - 1)
        {
            support_ptr[i] = 0.0f; // Not enough data points, return 0 or handle as needed
        }
        else
        {
            float lowest_low = low_ptr[i];
            for (int j = 0; j < lookback; ++j)
            {
                lowest_low = std::min(lowest_low, low_ptr[i - j]);
            }
            support_ptr[i] = lowest_low;
        }
    }

    return support;
}

// Define the module
PYBIND11_MODULE(ta_module, m)
{
    m.doc() = "Module for computing technical analysis indicators";
    m.def("computeEMA", &computeEMA, "Compute Exponential Moving Average (EMA)",
          pybind11::arg("data_array"), pybind11::arg("lookback"));
    m.def("computeStochasticOscillator", &computeStochasticOscillator, "Compute Stochastic Oscillator",
          pybind11::arg("close_prices"), pybind11::arg("low_prices"), pybind11::arg("high_prices"), pybind11::arg("lookback"));
    m.def("computeResistance", &computeResistance, "Compute Resistance Level",
          pybind11::arg("high_prices"), pybind11::arg("lookback"));
    m.def("computeSupport", &computeSupport, "Compute Support Level",
          pybind11::arg("low_prices"), pybind11::arg("lookback"));
}
