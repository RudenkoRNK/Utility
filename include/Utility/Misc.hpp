#pragma once

#define NOMINMAX

#include <algorithm>
#include <cassert>
#include <chrono>
#include <numeric>
#include <vector>

namespace Utility {

inline std::vector<size_t> GetIndices(size_t size) {
  auto indices = std::vector<size_t>(size);
  std::iota(indices.begin(), indices.end(), size_t{0});
  return indices;
}

template <typename T>
inline void Permute(std::vector<T> &v, std::vector<size_t> &perm) {
  Permute(v, perm, std::identity{});
}

template <typename T, typename Indexer, typename IndexFunction>
inline void Permute(std::vector<T> &v, std::vector<Indexer> &perm,
                    IndexFunction &&Index) {
#ifndef NDEBUG
  assert(std::unique(perm.begin(), perm.end(),
                     [&](Indexer const &lhs, Indexer const &rhs) {
                       return Index(lhs) == Index(rhs);
                     }) == perm.end());
  assert(*std::min_element(perm.begin(), perm.end(),
                           [&](Indexer const &lhs, Indexer const &rhs) {
                             return lhs < rhs;
                           }) == 0);
  assert(*std::max_element(perm.begin(), perm.end(),
                           [&](Indexer const &lhs, Indexer const &rhs) {
                             return lhs < rhs;
                           }) == perm.size() - 1);
  if constexpr (std::is_same_v<T, Indexer>) {
    assert(&v != &perm);
  }
#endif // !NDEBUG

  auto &&control = std::vector<size_t>(v.size());
  std::iota(control.begin(), control.end(), size_t{0});
  for (auto i = size_t{0}, e = v.size(); i < e; ++i) {
    while (Index(perm.at(i)) != i) {
      std::swap(control.at(i), control.at(Index(perm.at(i))));
      std::swap(perm.at(i), perm.at(Index(perm.at(i))));
    }
  }
  for (auto i = size_t{0}, e = v.size(); i < e; ++i) {
    while (control.at(i) != i) {
      std::swap(v.at(i), v.at(control.at(i)));
      std::swap(perm.at(i), perm.at(control.at(i)));
      std::swap(control.at(i), control.at(control.at(i)));
    }
  }
}

template <typename FG, typename... Args>
inline std::chrono::nanoseconds Benchmark(FG &&Func, Args &&... args) {
  auto start = std::chrono::steady_clock::now();
  Func(std::forward<Args>(args)...);
  auto end = std::chrono::steady_clock::now();
  return end - start;
}

} // namespace Utility
