#pragma once

#define NOMINMAX

#include <algorithm>
#include <cassert>
#include <chrono>
#include <exception>
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
  static_assert(std::is_nothrow_swappable_v<T>);
  static_assert(std::is_nothrow_swappable_v<Indexer>);
  static_assert(std::is_nothrow_invocable_v<IndexFunction, Indexer>);
  assert(v.size() == perm.size());
  if (v.size() == 0)
    return;
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
  static_assert(std::is_nothrow_move_constructible_v<T>);
  static_assert(std::is_nothrow_move_assignable_v<T>);
  T &restoreTo;
  T originalValue;

public:
  SaveRestore(T &value) : restoreTo(value), originalValue(value) {}
  SaveRestore(T &&value, T &restoreTo) noexcept
      : restoreTo(restoreTo), originalValue(std::move(value)) {}

  SaveRestore(SaveRestore const &) = delete;
  SaveRestore(SaveRestore &&) = delete;
  SaveRestore &operator=(SaveRestore const &) = delete;
  SaveRestore &operator=(SaveRestore &&) = delete;

  ~SaveRestore() { restoreTo = std::move(originalValue); }
};

template <typename T> class ExceptionGuard final {
  static_assert(noexcept(std::declval<T>().Clear()));
  T &value;

public:
  ExceptionGuard(T &value) noexcept : value(value) {}

  ExceptionGuard(ExceptionGuard const &) = delete;
  ExceptionGuard(ExceptionGuard &&) = delete;
  ExceptionGuard &operator=(ExceptionGuard const &) = delete;
  ExceptionGuard &operator=(ExceptionGuard &&) = delete;

  ~ExceptionGuard() {
    if (std::uncaught_exceptions())
      value.Clear();
  }
};

template <typename Container>
static size_t UnorderedHash(Container const &container) {
  auto hash = std::hash<typename Container::value_type>{};
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

template <class U> auto static IdWrapperNextId = std::atomic_size_t{0};
template <typename T> class IdWrapper final {
  T value;
  size_t id;

public:
  IdWrapper(T &&value) : value(std::move(value)), id(IdWrapperNextId<T> ++) {}

  size_t GetId() const noexcept { return id; }
  constexpr T const &Get() const noexcept { return value; }
  constexpr operator T const &() const noexcept { return value; }
};
template <class T>
static bool operator==(IdWrapper<T> const &lhs,
                       IdWrapper<T> const &rhs) noexcept {
  auto ieq = lhs.GetId() == rhs.GetId();
  assert((lhs.Get() == rhs.Get()) == ieq);
  return ieq;
}
template <class T> static size_t hash_value(IdWrapper<T> const &val) noexcept {
  return val.GetId();
}
} // namespace Utility

namespace std {
template <class T> struct hash<Utility::IdWrapper<T>> {
  size_t operator()(Utility::IdWrapper<T> const &val) const noexcept {
    return Utility::hash_value(val);
  }
};
} // namespace std
