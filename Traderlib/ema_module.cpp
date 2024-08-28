#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>

std::vector<float> computeEMA(const std::vector<float> &data, int span)
{
    if (span <= 0)
    {
        throw std::invalid_argument("Span must be a positive integer.");
    }

    if (data.empty())
    {
        return std::vector<float>();
    }

    std::vector<float> ema(data.size());
    float alpha = 2.0f / (span + 1);

    ema[0] = data[0];

    for (size_t i = 1; i < data.size(); ++i)
    {
        ema[i] = alpha * data[i] + (1.0f - alpha) * ema[i - 1];
    }

    return ema;
}

PYBIND11_MODULE(ema_module, m)
{
    m.def("computeEMA", &computeEMA, "A function that computes the Exponential Moving Average (EMA)",
          pybind11::arg("data"), pybind11::arg("span"));
}