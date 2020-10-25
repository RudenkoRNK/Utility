#pragma once

#define NOMINMAX

#include <algorithm>
#include <cassert>
#include <chrono>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace Utility {

static std::vector<size_t> GetIndices(size_t size) {
  auto indices = std::vector<size_t>(size);
  std::iota(indices.begin(), indices.end(), size_t{0});
  return indices;
}

template <typename T>
static void Permute(std::vector<T> &v, std::vector<size_t> &perm) {
  Permute(v, perm, std::identity{});
}

template <typename T, typename Indexer, typename IndexFunction>
static void Permute(std::vector<T> &v, std::vector<Indexer> &perm,
                    IndexFunction &&Index) {
#ifndef NDEBUG
  assert(std::unique(perm.begin(), perm.end(),
                     [&](Indexer const &lhs, Indexer const &rhs) {
                       return Index(lhs) == Index(rhs);
                     }) == perm.end());
  assert(Index(*std::min_element(perm.begin(), perm.end(),
                                 [&](Indexer const &lhs, Indexer const &rhs) {
                                   return lhs < rhs;
                                 })) == 0);
  assert(Index(*std::max_element(perm.begin(), perm.end(),
                                 [&](Indexer const &lhs, Indexer const &rhs) {
                                   return lhs < rhs;
                                 })) == perm.size() - 1);
  if constexpr (std::is_same_v<T, Indexer>) {
    assert(&v != &perm);
  }
  assert(v.size() == perm.size());
#endif // !NDEBUG

  auto &&control = std::vector<size_t>(v.size());
  std::iota(control.begin(), control.end(), size_t{0});
  for (auto i = size_t{0}, e = v.size(); i != e; ++i) {
    while (Index(perm[i]) != i) {
      std::swap(control[i], control[Index(perm[i])]);
      std::swap(perm[i], perm[Index(perm[i])]);
    }
  }
  for (auto i = size_t{0}, e = v.size(); i != e; ++i) {
    while (control[i] != i) {
      std::swap(v[i], v[control[i]]);
      std::swap(perm[i], perm[control[i]]);
      std::swap(control[i], control[control[i]]);
    }
  }
}

template <typename FG, typename... Args>
static std::chrono::nanoseconds Benchmark(FG &&Func, Args &&... args) {
  auto start = std::chrono::steady_clock::now();
  Func(std::forward<Args>(args)...);
  auto end = std::chrono::steady_clock::now();
  return end - start;
}

template <typename T> class SaveRestore final {
  static_assert(noexcept(T(std::declval<T>())));
  T originalValue;
  T &restoreTo;

public:
  SaveRestore(T &value) : restoreTo(value), originalValue(value) {}

  SaveRestore(T &&value, T &restoreTo)
      : restoreTo(restoreTo), originalValue(std::move(value)) {}

  SaveRestore(SaveRestore const &) = delete;
  SaveRestore(SaveRestore &&) = delete;
  SaveRestore &operator=(SaveRestore const &) = delete;
  SaveRestore &operator=(SaveRestore &&) = delete;

  ~SaveRestore() { restoreTo = std::move(originalValue); }
};

template <typename Container>
static size_t UnorderedHash(Container const &container) {
  auto hash = std::hash<Container::value_type>{};
  return std::accumulate(
      container.begin(), container.end(), size_t{1},
      [&](size_t cur_val, auto const &elem) { return cur_val * hash(elem); });
}

template <typename Container>
static auto Enumerate(Container const &container) {
  using ElementType = typename Container::value_type;
  auto map = std::unordered_map<ElementType, size_t>{};
  auto lastIndex = size_t{0};
  for (auto const &e : container) {
    if (map.contains(e))
      continue;
    map.emplace(e, lastIndex++);
  }
  return map;
}

template <typename T> class IdWrapper final {
  template <class U> auto static nextId = std::atomic_size_t{0};

  T value;
  size_t id;

public:
  IdWrapper(T &&value) : value(std::move(value)), id(nextId<T> ++) {}

  size_t GetId() const { return id; }
  constexpr T const &Get() const noexcept { return value; }
  constexpr operator T const &() const noexcept { return value; }
};

} // namespace Utility
