#pragma once

#include "Utility/TypeTraits.hpp"
#include <algorithm>
#include <cassert>
#include <complex>
#include <concepts>
#include <numeric>
#include <type_traits>
#include <vector>

namespace Utility {

template <typename Number> concept Numeric = requires() {
  requires std::is_default_constructible_v<Number>;
  requires std::is_arithmetic_v<Number> || isInstanceOf<std::complex, Number>;
  requires std::is_same_v<Number, std::remove_cvref_t<Number>>;
};

template <Numeric Number> struct LinearFit {
  // y = Const + Slope * x
  Number Const;
  Number Slope;
  Number SigmaConst;
  Number SigmaSlope;
};

template <Numeric Number> constexpr auto Mean(std::vector<Number> const &v) {
  auto sum = std::accumulate(v.begin(), v.end(), Number{});
  return sum / static_cast<double>(v.size());
}

template <Numeric Number>
constexpr auto Variance(std::vector<Number> const &v) {
  static_assert(!isInstanceOf<std::complex, Number>, "Not implemented");
  auto mean = Mean(v);
  using ContinuousType = decltype(mean);
  auto sum = std::inner_product(
      v.begin(), v.end(), v.begin(), ContinuousType{}, std::plus{},
      [&mean](auto x, auto y) { return (x - mean) * (y - mean); });
  return sum / static_cast<double>(v.size());
}

// Root mean square
template <Numeric Number> constexpr auto RMS(std::vector<Number> const &v) {
  static_assert(!isInstanceOf<std::complex, Number>, "Not implemented");
  auto sum = std::inner_product(v.begin(), v.end(), v.begin(), Number{});
  return std::sqrt(sum / static_cast<double>(v.size()));
}

template <Numeric NumberX, Numeric NumberY>
requires std::common_with<NumberX, NumberY> constexpr auto
LeastSquares(std::vector<NumberX> const &x, std::vector<NumberY> const &y) {
  assert(x.size() == y.size());
  assert(y.size() >= 2);
  auto n = y.size();
  auto xAvg = Mean(x);
  auto yAvg = Mean(y);
  using ContinuousNumber = decltype(xAvg + yAvg);
  auto x2Avg = std::pow(RMS(x), 2);
  auto y2Avg = std::pow(RMS(y), 2);
  auto xyAvg =
      std::inner_product(x.begin(), x.end(), y.begin(), ContinuousNumber{}) / n;
  // y = Const + Slope * x
  auto Slope = (xyAvg - xAvg * yAvg) / (x2Avg - xAvg * xAvg);
  auto Const = yAvg - Slope * xAvg;
  auto SigmaSlope =
      std::sqrt((y2Avg - yAvg * yAvg) / (x2Avg - xAvg * xAvg) - Slope * Slope) /
      std::sqrt(n);
  auto SigmaConst = SigmaSlope * sqrt(x2Avg - xAvg * xAvg);
  return LinearFit<ContinuousNumber>{.Const = Const,
                                     .Slope = Slope,
                                     .SigmaConst = SigmaConst,
                                     .SigmaSlope = SigmaSlope};
}

template <Numeric NumberY>
constexpr auto LeastSquares(std::vector<NumberY> const &y) {
  auto x = std::vector<double>(y.size());
  std::iota(x.begin(), x.end(), 0.0);
  return LeastSquares(x, y);
}

} // namespace Utility
