#pragma once

#include "Utility/TypeTraits.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <exception>
#include <numeric>
#include <random>
#include <vector>

namespace Utility {

inline std::vector<size_t> GetIndices(size_t size) {
  auto indices = std::vector<size_t>(size);
  std::iota(indices.begin(), indices.end(), size_t{0});
  return indices;
}

template <typename Vector, typename VectorIndexers, typename IndexFunction>
void Permute(Vector &v, VectorIndexers &perm, IndexFunction &&Index) {
  using T = typename Vector::value_type;
  using Indexer = typename VectorIndexers::value_type;
  static_assert(std::is_nothrow_swappable_v<T>);
  static_assert(std::is_nothrow_swappable_v<Indexer>);
  static_assert(std::is_nothrow_invocable_v<IndexFunction, Indexer>);
  using std::swap;
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
      swap(control[i], control[Index(perm[i])]);
      swap(perm[i], perm[Index(perm[i])]);
    }
  }
  for (auto i = size_t{0}, e = v.size(); i != e; ++i) {
    while (control[i] != i) {
      swap(v[i], v[control[i]]);
      swap(perm[i], perm[control[i]]);
      swap(control[i], control[control[i]]);
    }
  }
}

template <typename Vector> void Permute(Vector &v, std::vector<size_t> &perm) {
  Permute(v, perm, std::identity{});
}

template <typename Vector, typename Comparator>
std::vector<size_t> GetSortPermutation(Vector const &v, Comparator &&cmp) {
  auto permutation = Utility::GetIndices(v.size());
  std::sort(
      permutation.begin(), permutation.end(),
      [&](size_t index0, size_t index1) { return cmp(v[index0], v[index1]); });
  return permutation;
}

template <typename Vector>
std::vector<size_t> GetSortPermutation(Vector const &v) {
  return GetSortPermutation(v, std::less<>{});
}

template <typename Generator = std::mt19937>
constexpr Generator &GetRandomGenerator() {
  auto static thread_local generator = Generator{std::random_device{}()};
  return generator;
}

template <typename FG, typename... Args>
std::chrono::nanoseconds _Benchmark(FG &&Func, size_t nRuns, Args &&... args) {
  static_assert(CallableTraits<FG>::nArguments == sizeof...(args));
  auto start = std::chrono::steady_clock::now();
  for (auto i = size_t{0}; i < nRuns; ++i)
    Func(std::forward<Args>(args)...);
  auto end = std::chrono::steady_clock::now();
  return (end - start) / nRuns;
}

template <std::size_t... Indices>
auto _AppendSize(std::index_sequence<Indices...>)
    -> std::index_sequence<sizeof...(Indices), Indices...> {
  return {};
}

template <typename FG, typename Tuple, std::size_t... Indices>
std::chrono::nanoseconds _Benchmark2(FG &&Func, Tuple &&tuple,
                                     std::index_sequence<Indices...>) {
  return _Benchmark(std::forward<FG>(Func),
                    std::get<Indices>(std::forward<Tuple>(tuple))...);
}

template <typename FG, typename... Args>
std::chrono::nanoseconds Benchmark(FG &&Func, Args &&... args) {
  if constexpr (CallableTraits<FG>::nArguments + 1 == sizeof...(args)) {
    auto &&tuple = std::forward_as_tuple(std::forward<Args>(args)...);
    auto &&Inds =
        _AppendSize(std::make_index_sequence<CallableTraits<FG>::nArguments>{});
    return _Benchmark2(std::forward<FG>(Func),
                       std::forward<decltype(tuple)>(tuple), Inds);
  } else
    return _Benchmark(std::forward<FG>(Func), size_t{1},
                      std::forward<Args>(args)...);
}

// Similar to boost::logic::tribool but with other naming
class AutoOption final {
  enum class Option { False, True, Auto };
  Option option;

public:
  constexpr AutoOption() noexcept : option(Option::Auto){};
  constexpr AutoOption(bool option) noexcept
      : option(option ? Option::True : Option::False){};
  constexpr bool isTrue() const noexcept { return option == Option::True; }
  constexpr bool isFalse() const noexcept { return option == Option::False; }
  constexpr bool isAuto() const noexcept { return option == Option::Auto; }
  constexpr AutoOption operator!() const noexcept {
    if (isAuto())
      return AutoOption{};
    return AutoOption{isTrue() ? false : true};
  }
  explicit constexpr operator bool() const noexcept {
    return option == Option::True;
  }

  static constexpr AutoOption True() noexcept { return AutoOption{true}; }
  static constexpr AutoOption False() noexcept { return AutoOption{false}; }
  static constexpr AutoOption Auto() noexcept { return AutoOption{}; }
};
constexpr AutoOption operator&&(AutoOption x, AutoOption y) noexcept {
  if (x.isFalse() || y.isFalse())
    return AutoOption::False();
  if (x.isAuto() || y.isAuto())
    return AutoOption{};
  return AutoOption::True();
}
constexpr AutoOption operator||(AutoOption x, AutoOption y) noexcept {
  if (x.isTrue() || y.isTrue())
    return AutoOption::True();
  if (x.isAuto() || y.isAuto())
    return AutoOption::Auto();
  return AutoOption::False();
}
constexpr AutoOption operator==(AutoOption x, AutoOption y) noexcept {
  // Lukasiewicz logic
  if (x.isTrue() && y.isTrue())
    return AutoOption::True();
  if (x.isFalse() && y.isFalse())
    return AutoOption::True();
  if (x.isAuto() && y.isAuto())
    return AutoOption::True();
  if (x.isAuto() || y.isAuto())
    return AutoOption::Auto();
  return AutoOption::False();
}
constexpr AutoOption operator!=(AutoOption x, AutoOption y) noexcept {
  return !(x == y);
}

template <typename T> class SaveRestore final {
  static_assert(std::is_nothrow_move_assignable_v<T>);
  static_assert(!std::is_reference_v<T>);
  T &restoreTo;
  T originalValue;

public:
  explicit constexpr SaveRestore(T &value) noexcept(
      std::is_nothrow_copy_constructible_v<T>)
      : restoreTo(value), originalValue(std::as_const(value)) {}
  explicit constexpr SaveRestore(T &&value, T &restoreTo) noexcept(
      std::is_nothrow_move_constructible_v<T>)
      : restoreTo(restoreTo), originalValue(std::move(value)) {}

  SaveRestore(SaveRestore const &) = delete;
  SaveRestore(SaveRestore &&) = delete;
  SaveRestore &operator=(SaveRestore const &) = delete;
  SaveRestore &operator=(SaveRestore &&) = delete;

  ~SaveRestore() noexcept { restoreTo = std::move(originalValue); }
};

template <typename NoExceptionCallable, typename ExceptionCallable>
class RAII final {
  static_assert(noexcept(std::declval<ExceptionCallable>()()));
  static_assert(!std::is_reference_v<NoExceptionCallable>);
  NoExceptionCallable callNoException;
  ExceptionCallable callException;

public:
  RAII(NoExceptionCallable &&callAlways)
      : callNoException(std::move(callAlways)), callException(callNoException) {
    static_assert(std::is_reference_v<ExceptionCallable>);
  }

  RAII(NoExceptionCallable &&callNoException, ExceptionCallable &&callException)
      : callNoException(std::move(callNoException)),
        callException(std::move(callException)) {}

  RAII(RAII const &) = delete;
  RAII(RAII &&) = delete;
  RAII &operator=(RAII const &) = delete;
  RAII &operator=(RAII &&) = delete;

  ~RAII() noexcept(noexcept(std::declval<NoExceptionCallable>()())) {
    if (!std::uncaught_exceptions())
      callNoException();
    else
      callException();
  }
};
template <typename NoExceptionCallable>
RAII(NoExceptionCallable &&)
    ->RAII<std::remove_reference_t<NoExceptionCallable>,
           std::add_lvalue_reference_t<NoExceptionCallable>>;

// Save exceptions in multithreaded environment
class ExceptionSaver final {
  std::atomic<size_t> nCapturedExceptions;
  std::atomic<size_t> nSavedExceptions;
  std::vector<std::exception_ptr> exceptions;
  size_t maxExceptions;

public:
  ExceptionSaver(size_t maxExceptions = 1) : maxExceptions(maxExceptions) {
    exceptions.resize(maxExceptions);
  }

  template <typename Callable> auto Wrap(Callable &&callable) {
    using ReturnType = typename CallableTraits<Callable>::template Type<0>;
    static_assert(std::is_void_v<ReturnType> ||
                  (std::is_nothrow_default_constructible_v<ReturnType> &&
                   !std::is_reference_v<ReturnType>));
    return _Wrap(
        std::forward<Callable>(callable),
        std::make_index_sequence<CallableTraits<Callable>::nArguments>{});
  }

  ExceptionSaver(ExceptionSaver const &) = delete;
  ExceptionSaver(ExceptionSaver &&) = delete;
  ExceptionSaver &operator=(ExceptionSaver const &) = delete;
  ExceptionSaver &operator=(ExceptionSaver &&) = delete;

  // Not thread-safe
  void Rethrow() {
    if (!nSavedExceptions)
      return;
    auto e = exceptions[--nSavedExceptions];
    exceptions[nSavedExceptions] = std::exception_ptr{};
    std::rethrow_exception(e);
  }
  size_t NCapturedExceptions() const noexcept { return nCapturedExceptions; }
  size_t NSavedExceptions() const noexcept { return nSavedExceptions; }

private:
  template <class Callable, size_t... Indices>
  auto _Wrap(Callable &&callable, std::integer_sequence<size_t, Indices...>) {
    using ReturnType = typename CallableTraits<Callable>::template Type<0>;
    return [&](typename CallableTraits<Callable>::template ArgType<
               Indices>... args) {
      try {
        return callable(
            std::forward<
                typename CallableTraits<Callable>::template ArgType<Indices>>(
                args)...);
      } catch (std::exception &) {
        size_t index = nCapturedExceptions++;
        if (index < maxExceptions) {
          ++nSavedExceptions;
          exceptions[index] = std::current_exception();
        }
        if constexpr (!std::is_void_v<ReturnType>)
          return ReturnType{};
      }
    };
  }
};

template <typename Enum, Enum LastElement>
constexpr Enum &operator++(Enum &element) noexcept {
  auto max = static_cast<int>(LastElement);
  auto e = static_cast<int>(element);
  auto nextE = (e + 1) % (max + 1);
  element = static_cast<Enum>(nextE);
  return element;
}
template <typename Enum, Enum LastElement>
constexpr Enum operator++(Enum &element, int) noexcept {
  auto old = element;
  operator++<Enum, LastElement>(element);
  return old;
}
template <typename Enum, Enum LastElement>
constexpr Enum &operator--(Enum &element) noexcept {
  auto max = static_cast<int>(LastElement);
  auto e = static_cast<int>(element);
  auto nextE = (e + max) % (max + 1);
  element = static_cast<Enum>(nextE);
  return element;
}
template <typename Enum, Enum LastElement>
constexpr Enum operator--(Enum &element, int) noexcept {
  auto old = element;
  operator--<Enum, LastElement>(element);
  return old;
}

} // namespace Utility
