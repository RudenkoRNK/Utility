#pragma once

#define NOMINMAX

#include <Utility/TypeTraits.hpp>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <exception>
#include <execution>
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

template <typename FG, typename... Args>
static std::chrono::nanoseconds _Benchmark(FG &&Func, size_t nRuns,
                                           Args &&... args) {
  static_assert(CallableTraits<FG>::nArguments == sizeof...(args));
  auto start = std::chrono::steady_clock::now();
  for (auto i = size_t{0}; i < nRuns; ++i)
    Func(std::forward<Args>(args)...);
  auto end = std::chrono::steady_clock::now();
  return (end - start) / nRuns;
}

template <std::size_t... Indices>
static auto _AppendSize(std::index_sequence<Indices...>)
    -> std::index_sequence<sizeof...(Indices), Indices...> {
  return {};
}

template <typename FG, typename Tuple, std::size_t... Indices>
static auto _Benchmark2(FG &&Func, Tuple &&tuple,
                        std::index_sequence<Indices...>) {
  return _Benchmark(std::forward<FG>(Func),
                    std::get<Indices>(std::forward<Tuple>(tuple))...);
}

template <typename FG, typename... Args>
static std::chrono::nanoseconds Benchmark(FG &&Func, Args &&... args) {
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
static constexpr AutoOption operator&&(AutoOption x, AutoOption y) noexcept {
  if (x.isFalse() || y.isFalse())
    return AutoOption{false};
  if (x.isAuto() || y.isAuto())
    return AutoOption{};
  return AutoOption{true};
}
static constexpr AutoOption operator||(AutoOption x, AutoOption y) noexcept {
  if (x.isTrue() || y.isTrue())
    return AutoOption{true};
  if (x.isAuto() || y.isAuto())
    return AutoOption{};
  return AutoOption{false};
}
static constexpr AutoOption operator==(AutoOption x, AutoOption y) noexcept {
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
static constexpr AutoOption operator!=(AutoOption x, AutoOption y) noexcept {
  return !(x == y);
}

template <typename T> class SaveRestore final {
  static_assert(std::is_nothrow_move_constructible_v<T>);
  static_assert(std::is_nothrow_move_assignable_v<T>);
  T &restoreTo;
  T originalValue;

public:
  explicit SaveRestore(T &value) : restoreTo(value), originalValue(value) {}
  explicit SaveRestore(T &&value, T &restoreTo) noexcept
      : restoreTo(restoreTo), originalValue(std::move(value)) {}

  SaveRestore(SaveRestore const &) = delete;
  SaveRestore(SaveRestore &&) = delete;
  SaveRestore &operator=(SaveRestore const &) = delete;
  SaveRestore &operator=(SaveRestore &&) = delete;

  ~SaveRestore() { restoreTo = std::move(originalValue); }
};

template <typename NoExceptionCallable, typename ExceptionCallable>
class RAII final {
  static_assert(noexcept(std::declval<ExceptionCallable>()()));
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

  ~RAII() {
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
  size_t GetNCapturedExceptions() noexcept { return nCapturedExceptions; }
  size_t GetNSavedExceptions() noexcept { return nSavedExceptions; }

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

template <typename T> class IdWrapper final {
  T value;
  size_t id;

public:
  IdWrapper(T &&value, size_t id) : value(std::move(value)), id(id) {}

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
