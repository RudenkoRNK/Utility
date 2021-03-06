#pragma once

#include "utility/type_traits.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <concepts>
#include <exception>
#include <numeric>
#include <optional>
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
  for (auto i = 0; i != nRuns; ++i)
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
  std::optional<T> originalValue;
  T *restoreTo = nullptr;

public:
  constexpr SaveRestore() noexcept {};
  explicit SaveRestore(T &value) noexcept(
      std::is_nothrow_copy_constructible_v<T>)
      : originalValue{std::as_const(value)}, restoreTo{&value} {}
  explicit SaveRestore(T &&value) noexcept(
      std::is_nothrow_move_constructible_v<T>)
      : originalValue{std::move(value)}, restoreTo{&value} {}
  explicit SaveRestore(T &&value, T &restoreTo) noexcept(
      std::is_nothrow_move_constructible_v<T>)
      : originalValue{std::move(value)}, restoreTo{&restoreTo} {}

  SaveRestore(SaveRestore const &) = delete;
  SaveRestore(SaveRestore &&other) noexcept { swap(other); }
  SaveRestore &operator=(SaveRestore const &) = delete;
  SaveRestore &operator=(SaveRestore &&other) &noexcept {
    swap(other);
    return *this;
  }

  void swap(SaveRestore &other) noexcept {
    static_assert(std::is_nothrow_swappable_v<T>);
    std::swap(restoreTo, other.restoreTo);
    std::swap(originalValue, other.originalValue);
  }

  ~SaveRestore() noexcept {
    if (!restoreTo)
      return;
    *restoreTo = std::move(originalValue.value());
  }
};

struct CallAlways final {};
struct CallOnException final {};

template <std::invocable Callable,
          std::invocable CallableOnException = Callable>
class RAII final {
  std::optional<Callable> callNormally;
  std::optional<CallableOnException> callOnException;

public:
  RAII() noexcept {};
  RAII(Callable &&callable, CallAlways = CallAlways{})
      : callNormally(std::move(callable)) {
    static_assert(std::is_nothrow_invocable_v<Callable>);
  }
  RAII(Callable &&callable, CallOnException)
      : callOnException(std::move(callable)) {
    static_assert(std::is_nothrow_invocable_v<Callable>);
  }
  RAII(Callable &&callNormally, CallableOnException &&callOnException)
      : callNormally(std::forward<Callable>(callNormally)),
        callOnException(std::forward<CallableOnException>(callOnException)) {
    static_assert(std::is_nothrow_invocable_v<CallableOnException>);
  }

  RAII(RAII const &) = delete;
  RAII(RAII &&other) noexcept { swap(other); }
  RAII &operator=(RAII const &) = delete;
  RAII &operator=(RAII &&other) &noexcept {
    swap(other);
    return *this;
  }

  void swap(RAII &other) noexcept {
    static_assert(std::is_nothrow_swappable_v<Callable>);
    static_assert(std::is_nothrow_swappable_v<CallableOnException>);
    callNormally.swap(other.callNormally);
    callOnException.swap(other.callOnException);
  }

  ~RAII() noexcept(std::is_nothrow_invocable_v<Callable>) {
    if (!std::uncaught_exceptions())
      CallNormally();
    else
      CallOnException();
  }

private:
  void CallNormally() noexcept(std::is_nothrow_invocable_v<Callable>) {
    if (callNormally)
      callNormally.value()();
  }
  void CallOnException() noexcept {
    if (callOnException) {
      assert(std::is_nothrow_invocable_v<CallableOnException>);
      callOnException.value()();
    } else if (callNormally) {
      assert(std::is_nothrow_invocable_v<Callable>);
      callNormally.value()();
    }
  }
};

// Save exceptions in multithreaded environment
class ExceptionSaver final {
  std::atomic<size_t> nCapturedExceptions = 0;
  std::atomic<size_t> nSavedExceptions = 0;
  std::vector<std::exception_ptr> exceptions;

public:
  ExceptionSaver(size_t maxExceptions = 1) { exceptions.resize(maxExceptions); }
  ExceptionSaver(ExceptionSaver const &) = delete;
  ExceptionSaver(ExceptionSaver &&other) noexcept { swap(other); }
  ExceptionSaver &operator=(ExceptionSaver const &) = delete;
  ExceptionSaver &operator=(ExceptionSaver &&other) &noexcept {
    swap(other);
    return *this;
  }
  ~ExceptionSaver() { Rethrow(); }

  size_t NCapturedExceptions() const noexcept { return nCapturedExceptions; }
  size_t NSavedExceptions() const noexcept { return nSavedExceptions; }

  void swap(ExceptionSaver &other) noexcept {
    nCapturedExceptions =
        other.nCapturedExceptions.exchange(nCapturedExceptions);
    nSavedExceptions = other.nSavedExceptions.exchange(nSavedExceptions);
    std::swap(exceptions, other.exceptions);
  }

  // Wraps callable in a thread-save wrapper
  template <typename Callable> auto Wrap(Callable &&callable) {
    using ReturnType = typename CallableTraits<Callable>::template Type<0>;
    static_assert(std::is_void_v<ReturnType> ||
                  (std::is_nothrow_default_constructible_v<ReturnType> &&
                   !std::is_reference_v<ReturnType>));
    return _Wrap(
        std::forward<Callable>(callable),
        std::make_index_sequence<CallableTraits<Callable>::nArguments>{});
  }

  void Rethrow() {
#ifndef NDEBUG
    for (auto i = size_t{0}, e = exceptions.size(); i != e; ++i)
      assert(static_cast<bool>(exceptions[i]) == (i < nSavedExceptions));
#endif // !NDEBUG
    if (!nSavedExceptions)
      return;
    auto e = std::exception_ptr{};
    std::swap(e, exceptions[--nSavedExceptions]);
    std::rethrow_exception(e);
  }
  void Drop() noexcept {
    std::fill_n(exceptions.begin(), nSavedExceptions.load(),
                std::exception_ptr{});
    nSavedExceptions = 0;
#ifndef NDEBUG
    for (auto &&ptr : exceptions)
      assert(!ptr);
#endif // !NDEBUG
  }
  void SetMaxExceptions(size_t maxExceptions) {
    exceptions.resize(maxExceptions);
  }

private:
  template <class Callable, size_t... Indices>
  auto _Wrap(Callable &&callable, std::integer_sequence<size_t, Indices...>) {
    using ReturnType = typename CallableTraits<Callable>::template Type<0>;
    return [&](typename CallableTraits<Callable>::template ArgType<
               Indices>... args) noexcept {
      try {
        return callable(
            std::forward<
                typename CallableTraits<Callable>::template ArgType<Indices>>(
                args)...);
      } catch (...) {
        size_t index = nCapturedExceptions++;
        if (index < exceptions.size()) {
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
