#pragma once
#define NOMINMAX
#include <algorithm>
#include <cassert>
#include <chrono>
#include <numeric>
#include <vector>

namespace Utility {

struct LinearFit {
  // y = Const + Slope * x
  double Const;
  double Slope;
  double SigmaConst;
  double SigmaSlope;
};

template <class Number> double Mean(std::vector<Number> const &v) {
  auto sum = std::accumulate(v.begin(), v.end(), 0.0);
  return sum / v.size();
}

template <class Number> double Variance(std::vector<Number> const &v) {
  auto mean = Mean(v);
  auto sum = std::accumulate(v.begin(), v.end(), 0.0, [mean](auto x, auto y) {
    return x + (y - mean) * (y - mean);
  });
  return sum / v.size();
}

// Root mean square
template <class Number> double RMS(std::vector<Number> const &v) {
  auto sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
  return sqrt(sum / v.size());
}

template <class NumberX, class NumberY>
LinearFit LeastSquares(std::vector<NumberX> const &x,
                       std::vector<NumberY> const &y) {
  assert(x.size() == y.size());
  assert(y.size() >= 2);
  auto n = y.size();
  auto xAvg = Mean(x);
  auto yAvg = Mean(y);
  auto x2Avg = pow(RMS(x), 2);
  auto y2Avg = pow(RMS(y), 2);
  auto xyAvg = std::inner_product(x.begin(), x.end(), y.begin(), 0.0) / n;
  // y = Const + Slope * x
  auto Slope = (xyAvg - xAvg * yAvg) / (x2Avg - xAvg * xAvg);
  auto Const = yAvg - Slope * xAvg;
  auto SigmaSlope =
      sqrt((y2Avg - yAvg * yAvg) / (x2Avg - xAvg * xAvg) - Slope * Slope) /
      sqrt(n);
  auto SigmaConst = SigmaSlope * sqrt(x2Avg - xAvg * xAvg);
  return {.Const = Const,
          .Slope = Slope,
          .SigmaConst = SigmaConst,
          .SigmaSlope = SigmaSlope};
}

template <class Number> LinearFit LeastSquares(std::vector<Number> const &y) {
  auto x = std::vector<double>(y.size());
  std::iota(x.begin(), x.end(), 0.0);
  return LeastSquares(x, y);
}

} // namespace Utility
