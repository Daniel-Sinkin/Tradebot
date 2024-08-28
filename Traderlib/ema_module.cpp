#include <pybind11/pybind11.h>
#include <pybind11/stl.h>      // Needed to convert std::vector to Python list
#include <pybind11/numpy.h>    // Needed to handle NumPy arrays
#include <vector>
#include <stdexcept>

std::vector<float> computeEMA(const pybind11::array_t<float> &data_array, int span)
{
    if (span <= 0)
    {
        throw std::invalid_argument("Span must be a positive integer.");
    }

    // Get the buffer info from the NumPy array
    pybind11::buffer_info buf = data_array.request();

    // Ensure the input is a 1D array
    if (buf.ndim != 1)
    {
        throw std::runtime_error("Input data should be a 1D array.");
    }

    // Create a pointer to the data
    const float *data_ptr = static_cast<float *>(buf.ptr);
    size_t size = buf.size;

    if (size == 0)
    {
        return std::vector<float>();
    }

    std::vector<float> ema(size);
    float alpha = 2.0f / (span + 1);

    ema[0] = data_ptr[0];

    for (size_t i = 1; i < size; ++i)
    {
        ema[i] = alpha * data_ptr[i] + (1.0f - alpha) * ema[i - 1];
    }

    return ema;
}

PYBIND11_MODULE(ema_module, m)
{
    m.def("computeEMA", &computeEMA, "A function that computes the Exponential Moving Average (EMA)",
          pybind11::arg("data"), pybind11::arg("span"));
}