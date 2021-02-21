#define BOOST_TEST_MODULE Test

#include "Utility/Math.hpp"
#include "Utility/Misc.hpp"
#include "Utility/TypeTraits.hpp"
#include <array>
#include <boost/test/included/unit_test.hpp>
#include <complex>
#include <execution>
#include <functional>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>

int foo(int x) { return x; };
void bar(int x){};

BOOST_AUTO_TEST_CASE(arg_traits_test) {
  auto lambda1 = [](std::string const &) { return 0; };
  using T = Utility::CallableTraits<decltype(lambda1)>::Type<1>;
  // NOLINTNEXTLINE
  auto lambda2 = [&](T t) { return lambda1(t); };

  BOOST_TEST(Utility::CallableTraits<decltype(lambda1)>::isConst<1>);
  BOOST_TEST(Utility::CallableTraits<decltype(lambda1)>::isLValueReference<1>);
  BOOST_TEST(Utility::CallableTraits<decltype(lambda2)>::isConst<1>);
  BOOST_TEST(Utility::CallableTraits<decltype(lambda2)>::isLValueReference<1>);
  BOOST_TEST(Utility::CallableTraits<decltype(foo)>::isValue<0>);
  BOOST_TEST(Utility::CallableTraits<decltype(foo)>::isValue<1>);
  BOOST_TEST(!Utility::CallableTraits<decltype(bar)>::isValue<0>);
}

BOOST_AUTO_TEST_CASE(arg_traits_test_2) {
  auto x = 0;
  // NOLINTNEXTLINE
  auto lambda1 = [x](std::string const &) mutable {
    ++x;
    return 0;
  };
  // NOLINTNEXTLINE
  auto const lambda2 = [x](std::string const &) mutable {
    ++x;
    return 0;
  };
  // NOLINTNEXTLINE
  auto lambda3 = [&](std::string const &) {
    ++x;
    return 0;
  };
  // NOLINTNEXTLINE
  auto const lambda4 = [&](std::string const &) {
    ++x;
    return 0;
  };

  struct AAA {
    void X() const {};
    void Y(){};
  };

  auto func1 = std::function(lambda1);
  auto func2 = std::function(lambda2);
  auto func3 = std::function(lambda3);
  auto func4 = std::function(lambda4);

  BOOST_TEST(!Utility::CallableTraits<decltype(lambda1)>::isCallableConst);
  BOOST_TEST(!Utility::CallableTraits<decltype(lambda2)>::isCallableConst);
  BOOST_TEST(Utility::CallableTraits<decltype(lambda3)>::isCallableConst);
  BOOST_TEST(Utility::CallableTraits<decltype(lambda4)>::isCallableConst);
  BOOST_TEST(Utility::CallableTraits<decltype(func1)>::isCallableConst);
  BOOST_TEST(Utility::CallableTraits<decltype(func2)>::isCallableConst);
  BOOST_TEST(Utility::CallableTraits<decltype(func3)>::isCallableConst);
  BOOST_TEST(Utility::CallableTraits<decltype(func4)>::isCallableConst);
  BOOST_TEST(Utility::CallableTraits<decltype(&AAA::X)>::isCallableConst);
  BOOST_TEST(!Utility::CallableTraits<decltype(&AAA::Y)>::isCallableConst);
  BOOST_TEST(Utility::CallableTraits<decltype(foo)>::isCallableConst);

  BOOST_TEST(Utility::CallableTraits<
                 std::add_rvalue_reference_t<decltype(lambda1)>>::nArguments ==
             1);
  BOOST_TEST(Utility::CallableTraits<
                 std::remove_cvref_t<decltype(lambda1)>>::nArguments == 1);
}

BOOST_AUTO_TEST_CASE(arg_traits_test_3) {
  auto lambda1 = [](std::string const &, int, double, int, char) -> int {
    return 0;
  };
  using U1 = typename Utility::CallableTraits<decltype(lambda1)>::std_function;
  using T1 = std::function<int(std::string const &, int, double, int, char)>;
  auto lambda2 = [](int, double) {};
  using T2 = std::function<void(int, double)>;
  using U2 = typename Utility::CallableTraits<decltype(lambda2)>::std_function;
  BOOST_TEST((std::is_same_v<U1, T1>));
  BOOST_TEST((std::is_same_v<U2, T2>));
}

BOOST_AUTO_TEST_CASE(type_traits_test) {
  BOOST_TEST((Utility::isInstanceOf<std::vector, std::vector<int>>));
  BOOST_TEST((!Utility::isInstanceOf<std::vector, std::unordered_set<int>>));
  BOOST_TEST((!Utility::isInstanceOf<std::vector, int>));
}

BOOST_AUTO_TEST_CASE(perm_test) {
  auto perm = std::vector<size_t>{5, 2, 3, 0, 1, 4};
  auto perm2 = std::vector<size_t>{5, 2, 3, 0, 1, 4};
  auto v = std::vector<size_t>(perm.size());
  std::iota(v.begin(), v.end(), 0);
  Utility::Permute(v, perm);
  BOOST_TEST(v == perm);
  BOOST_TEST(v == perm2);
}

BOOST_AUTO_TEST_CASE(utility_test) {
  auto x = std::vector<size_t>{0, 1, 2, 3, 4};
  BOOST_TEST(abs(Utility::Mean(x) - 2) < 0.000001);
  BOOST_TEST(abs(Utility::RMS(x) - sqrt(6)) < 0.000001);
  BOOST_TEST(abs(Utility::Variance(x) - 2) < 0.000001);
  auto fit = Utility::LeastSquares(x);
  BOOST_TEST(abs(fit.Slope - 1) < 0.000001);

  auto N = 10001;
  auto v = Utility::GetIndices(N);
  BOOST_TEST(abs(Utility::Mean(v) - (N - 1) / 2) < 0.000001);
  BOOST_TEST(abs(Utility::RMS(v) - sqrt((N - 1) * (2 * N - 1) / 6)) < 0.0001);
  BOOST_TEST(abs(Utility::Variance(v) - (N * N - 1) / 12) < 0.000001);

  auto y = std::vector<std::complex<int>>{0, 1, 2, 3, 4};
  BOOST_TEST(abs(Utility::Mean(x) - 2) < 0.000001);
  BOOST_TEST(abs(Utility::RMS(x) - sqrt(6)) < 0.000001);
  BOOST_TEST(abs(Utility::Variance(x) - 2) < 0.000001);
  auto fit2 = Utility::LeastSquares(x);
  BOOST_TEST(abs(fit2.Slope - 1) < 0.000001);
}

BOOST_AUTO_TEST_CASE(save_restore) {
  auto perm = std::vector<size_t>{5, 2, 3, 0, 1, 4};
  auto checkperm = std::vector<size_t>{5, 2, 3, 0, 1, 4};
  {
    auto save = Utility::SaveRestore(perm);
    perm = std::vector<size_t>{};
  }
  BOOST_TEST(checkperm == perm);

  {
    auto save = Utility::SaveRestore(std::move(perm), perm);
    perm = std::vector<size_t>{};
  }
  BOOST_TEST(checkperm == perm);

  {
    auto save =
        Utility::SaveRestore(std::vector<size_t>{5, 2, 3, 0, 1, 4}, perm);
    perm = std::vector<size_t>{};
  }
  BOOST_TEST(checkperm == perm);
  {
    auto save = Utility::SaveRestore(std::move(perm));
    perm = std::vector<size_t>{};
  }
  BOOST_TEST(checkperm == perm);

  auto i = 123456;
  auto checki = i;
  {
    auto save = Utility::SaveRestore{i};
    i = 100;
  }
  BOOST_TEST(checki = i);

  { auto save = Utility::SaveRestore<int>{}; }
  {
    auto save = Utility::SaveRestore{i};
    i = 100;
    auto b = std::move(save);
  }
  BOOST_TEST(checki = i);
  {
    auto b = Utility::SaveRestore<int>{};
    {
      auto save = Utility::SaveRestore{i};
      i = 100;
      b = std::move(save);
    }
    BOOST_TEST(i == 100);
  }
  BOOST_TEST(checki = i);
}

namespace std {
template <> struct hash<std::pair<const int, double>> {
  size_t operator()(std::pair<const int, double> const &x) const noexcept {
    return std::hash<int>{}(x.first) * std::hash<double>{}(x.second);
  }
};
} // namespace std

struct AAA {
  int x;
  int y;
  AAA() = delete;
  AAA(int x, int y) : x(x), y(y){};
};
static bool operator==(AAA const &lhs, AAA const &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y;
}

BOOST_AUTO_TEST_CASE(exception_guard_test) {
  struct ThrowingStruct {
    int val = 1;
    void ThrowingMethod1() {
      auto g = Utility::RAII([]() {}, [&]() noexcept { Clear(); });
      if (val == 10)
        throw std::runtime_error("");
    }
    void Clear() noexcept { val = 0; }
  };
  auto g1 = Utility::RAII([]() noexcept {});
  auto s = ThrowingStruct{};
  s.ThrowingMethod1();
  BOOST_TEST(s.val == 1);
  s.val = 10;
  try {
    s.ThrowingMethod1();
  } catch (...) {
  }
  BOOST_TEST(s.val == 0);
}

BOOST_AUTO_TEST_CASE(benchmark_test) {
  struct AAA {
    AAA(size_t) {}
    AAA(AAA const &) = delete;
    AAA(AAA &&) = delete;
    AAA &operator=(AAA const &) = delete;
    AAA &operator=(AAA &&) = delete;
  };
  auto aaa = AAA(size_t{3});

  auto f1 = []() {};
  auto f2 = [](int x) {};
  auto f3 = [](int x, int y) {};
  auto f4 = [](size_t x) {};
  auto f5 = [](size_t x, size_t y) {};
  auto f6 = [](AAA &&x, size_t y) {};
  auto f7 = [](AAA &x, size_t y) {};
  auto f8 = [](AAA const &x, size_t y) {};
  Utility::Benchmark(f1);
  Utility::Benchmark(f1, size_t{10});
  Utility::Benchmark(f2, 3);
  Utility::Benchmark(f2, 3, size_t{10});
  Utility::Benchmark(f3, 3, 4);
  Utility::Benchmark(f3, 3, 4, size_t{10});
  Utility::Benchmark(f4, 3);
  Utility::Benchmark(f4, 3, size_t{10});
  Utility::Benchmark(f5, 3, size_t{4});
  Utility::Benchmark(f5, 3, size_t{4}, size_t{10});
  Utility::Benchmark(f6, AAA{3}, size_t{4});
  Utility::Benchmark(f6, AAA{3}, size_t{4}, size_t{10});
  Utility::Benchmark(f7, aaa, size_t{4});
  Utility::Benchmark(f7, aaa, size_t{4}, size_t{10});
  Utility::Benchmark(f8, AAA{3}, size_t{4});
  Utility::Benchmark(f8, AAA{3}, size_t{4}, size_t{10});
}

BOOST_AUTO_TEST_CASE(auto_option_test) {
  using namespace Utility;
  auto t = AutoOption::True();
  auto f = AutoOption::False();
  auto a = AutoOption::Auto();
  BOOST_TEST(t.isTrue());
  BOOST_TEST(f.isFalse());
  BOOST_TEST(!a.isTrue());
  BOOST_TEST(!a.isFalse());
  BOOST_TEST(a.isAuto());
  BOOST_TEST(static_cast<bool>(t == t));
  BOOST_TEST(static_cast<bool>(f == f));
  BOOST_TEST(static_cast<bool>(a == a));

  BOOST_TEST(static_cast<bool>(t == !f));
  BOOST_TEST(static_cast<bool>(f == !t));
  BOOST_TEST(static_cast<bool>(a == !a));
  BOOST_TEST(static_cast<bool>(t == !!t));
  BOOST_TEST(static_cast<bool>(f == !!f));
  BOOST_TEST(static_cast<bool>(a == !!a));

  BOOST_TEST(static_cast<bool>(t != f));
  BOOST_TEST(static_cast<bool>(f != t));
  BOOST_TEST(!static_cast<bool>(t != a));
  BOOST_TEST(!static_cast<bool>(f != a));

  BOOST_TEST(static_cast<bool>(t == true));
  BOOST_TEST(static_cast<bool>(f == false));
  BOOST_TEST(static_cast<bool>(t != false));
  BOOST_TEST(static_cast<bool>(f != true));
  BOOST_TEST(!static_cast<bool>(a != true));
  BOOST_TEST(!static_cast<bool>(a != false));

  BOOST_TEST(static_cast<bool>(t));
  BOOST_TEST(static_cast<bool>(!f));
  BOOST_TEST(!static_cast<bool>(a));
  BOOST_TEST(!static_cast<bool>(!a));
  BOOST_TEST(!static_cast<bool>(!!a));

  auto Impl = [](AutoOption x, AutoOption y) {
    if (x.isFalse() || y.isTrue())
      return AutoOption::True();
    if (x.isAuto() && y.isAuto())
      return AutoOption::True();
    if (x.isTrue() && y.isFalse())
      return AutoOption::False();
    return AutoOption::Auto();
  };
  auto Or = [&](AutoOption x, AutoOption y) { return Impl(Impl(x, y), y); };
  auto And = [&](AutoOption x, AutoOption y) { return !(Or(!x, !y)); };
  auto Eq = [&](AutoOption x, AutoOption y) {
    return And(Impl(x, y), Impl(y, x));
  };

  auto X = std::vector<AutoOption>{t, f, a};
  for (auto x : X)
    for (auto y : X) {
      BOOST_TEST(static_cast<bool>((x || y) == Or(x, y)));
      BOOST_TEST(static_cast<bool>((x && y) == And(x, y)));
      BOOST_TEST(static_cast<bool>((x == y) == Eq(x, y)));
    }

  if (t)
    BOOST_TEST(true);
  else
    BOOST_TEST(false);
  if (f)
    BOOST_TEST(false);
  else
    BOOST_TEST(true);
  if (a)
    BOOST_TEST(false);
  else
    BOOST_TEST(true);
  if (!a)
    BOOST_TEST(false);
  else
    BOOST_TEST(true);
}

BOOST_AUTO_TEST_CASE(exception_handler_test) {
  auto h = Utility::ExceptionSaver{10};
  auto i1 = Utility::GetIndices(20);

  std::for_each(std::execution::par_unseq, i1.begin(), i1.end(),
                h.Wrap([&](size_t i) { throw std::runtime_error{""}; }));

  while (h.NSavedExceptions()) {
    try {
      h.Rethrow();
    } catch (std::runtime_error &) {
    }
  }
  h.Rethrow();
}

BOOST_AUTO_TEST_CASE(raii_test) {
  auto copyCnt = size_t{0};
  auto noex = false;
  auto ex = false;
  struct Act {
    size_t &copyCnt;
    bool &act;
    Act(size_t &copyCnt, bool &act) : copyCnt(copyCnt), act(act) {}
    Act(Act const &a) : copyCnt(a.copyCnt), act(a.act) { ++copyCnt; }
    Act(Act &&a) : copyCnt(a.copyCnt), act(a.act) {}
    Act &operator=(Act const &a) { ++copyCnt; }
    Act &operator=(Act &&) {}
    void operator()() noexcept { act = true; }
  };

  {
    copyCnt = size_t{0};
    noex = false;
    ex = false;
    auto exact = Act(copyCnt, ex);
    auto noexact = Act(copyCnt, noex);
    auto raii = Utility::RAII(std::move(noexact), std::move(exact));
  }
  BOOST_TEST(noex == true);
  BOOST_TEST(ex == false);
  BOOST_TEST(copyCnt == 0);

  {
    copyCnt = size_t{0};
    noex = false;
    ex = false;
    auto raii = Utility::RAII(Act(copyCnt, noex), Act(copyCnt, ex));
  }
  BOOST_TEST(noex == true);
  BOOST_TEST(ex == false);
  BOOST_TEST(copyCnt == 0);

  {
    copyCnt = size_t{0};
    noex = false;
    ex = false;
    auto raii = Utility::RAII(Act(copyCnt, noex));
  }
  BOOST_TEST(noex == true);
  BOOST_TEST(ex == false);
  BOOST_TEST(copyCnt == 0);

  try {
    copyCnt = size_t{0};
    noex = false;
    ex = false;
    auto raii = Utility::RAII(Act(copyCnt, noex), Act(copyCnt, ex));
    throw std::runtime_error("");
  } catch (std::exception &) {
  }
  BOOST_TEST(noex == false);
  BOOST_TEST(ex == true);
  BOOST_TEST(copyCnt == 0);

  try {
    copyCnt = size_t{0};
    noex = false;
    ex = false;
    auto raii = Utility::RAII(Act(copyCnt, ex));
    throw std::runtime_error("");
  } catch (std::exception &) {
  }
  BOOST_TEST(noex == false);
  BOOST_TEST(ex == true);
  BOOST_TEST(copyCnt == 0);

  struct Restore {
    int *restoreTo;
    int value;
    Restore(int &restoreTo, int value) : restoreTo{&restoreTo}, value{value} {}
    void operator()() noexcept { *restoreTo = value; }
  };

  auto i = 123456;
  auto checki = i;
  {
    i = 100;
    auto restore = std::function([&]() noexcept { i = checki; });
    auto g = Utility::RAII<Restore>{};
    {
      auto r = Utility::RAII{Restore{i, checki}};
      r = std::move(g);
    }
    BOOST_TEST(i == 100);
    auto x = std::move(g);
  }
  BOOST_TEST(i == checki);

  auto r1 = Utility::RAII{Restore{i, checki}};
  auto r2 = Utility::RAII{Restore{i, checki}};
  r1.swap(r2);

  {
    i = 100;
    auto r =
        Utility::RAII{[&]() noexcept { i = checki; }, Utility::CallAlways{}};
  }
  BOOST_TEST(i == checki);

  {
    i = 100;
    auto r = Utility::RAII{[&]() noexcept { i = checki; },
                           Utility::CallOnException{}};
  }
  BOOST_TEST(i == 100);
  i = checki;

  try {
    i = 100;
    auto r = Utility::RAII{[&]() noexcept { i = checki; },
                           Utility::CallOnException{}};
    throw std::exception{};
  } catch (...) {
  }
  BOOST_TEST(i == checki);

  try {
    i = 100;
    auto r = Utility::RAII{[&]() {
                             i = checki;
                             throw std::exception{};
                           },
                           [&]() noexcept {}};
  } catch (...) {
  }
  BOOST_TEST(i == checki);
}

BOOST_AUTO_TEST_CASE(type_traits_forward_test) {
  auto F = [](std::string a, std::string &b, std::string &&c,
              std::string const &d, std::string const e,
              std::string const &&f) {};

  auto a = std::string("a");
  auto b = std::string("b");
  auto c = std::string("c");
  auto d = std::string("d");
  auto e = std::string("e");
  auto f = std::string("f");

  using T = Utility::CallableTraits<decltype(F)>;

  F(T::Forward<1>(a), T::Forward<2>(b), T::Forward<3>(c), T::Forward<4>(d),
    T::Forward<5>(e), T::Forward<6>(f));
  BOOST_TEST(a == "");
  BOOST_TEST(b == "b");
  BOOST_TEST(c == "c");
  BOOST_TEST(d == "d");
  BOOST_TEST(e == "");
  BOOST_TEST(f == "f");
}

BOOST_AUTO_TEST_CASE(sort_test) {
  auto x = std::vector<int>{5, 2, 1, 4, 3};
  auto sorted = x;
  std::sort(sorted.begin(), sorted.end());
  auto p = Utility::GetSortPermutation(x);
  Utility::Permute(x, p);
  BOOST_TEST(x == sorted);

  Utility::Permute(
      x, p, [](size_t const &i) noexcept { return i; });
}

BOOST_AUTO_TEST_CASE(random_test) {
  auto N = 1000;
  auto timen = std::chrono::nanoseconds{0};
  auto d = 0.0;
  auto rd = std::random_device{};

  auto rdtime = Utility::Benchmark([&]() { d += rd(); }, N);
  auto gentime = Utility::Benchmark(
      [&]() {
        auto gen = std::mt19937{};
        d += gen();
      },
      N);
  auto staticgentime = Utility::Benchmark(
      [&]() {
        auto &&gen = Utility::GetRandomGenerator();
        d += gen();
      },
      N);
  BOOST_TEST((gentime / staticgentime > 10));
  auto v = std::vector<int>{};
  std::generate(std::execution::par_unseq, v.begin(), v.end(), []() {
    auto &&gen = Utility::GetRandomGenerator();
    auto rand = std::uniform_int_distribution<int>(1);
    return rand(gen);
  });
}

BOOST_AUTO_TEST_CASE(enum_iterator_test) {
  using namespace Utility;
  enum class SomeEnum { A, B, C, D, E, F };
  auto h1 = SomeEnum::A;
  auto h2 = operator++<SomeEnum, SomeEnum::F>(h1);
  BOOST_TEST((h1 == SomeEnum::B));
  BOOST_TEST((h2 == SomeEnum::B));
  h1 = SomeEnum::F;
  h2 = operator++<SomeEnum, SomeEnum::F>(h1, 0);
  BOOST_TEST((h1 == SomeEnum::A));
  BOOST_TEST((h2 == SomeEnum::F));
  h1 = SomeEnum::A;
  h2 = operator--<SomeEnum, SomeEnum::F>(h1);
  BOOST_TEST((h1 == SomeEnum::F));
  BOOST_TEST((h2 == SomeEnum::F));
  h1 = SomeEnum::F;
  h2 = operator--<SomeEnum, SomeEnum::F>(h1, 0);
  BOOST_TEST((h1 == SomeEnum::E));
  BOOST_TEST((h2 == SomeEnum::F));

  auto cnt = 0;
  for (auto h = SomeEnum::F;
       operator++<SomeEnum, SomeEnum::F>(h) != SomeEnum::F;)
    ++cnt;
  BOOST_TEST(cnt == static_cast<int>(SomeEnum::F));
  cnt = 0;
  for (auto h = SomeEnum::F;
       operator--<SomeEnum, SomeEnum::F>(h) != SomeEnum::F;)
    ++cnt;
  BOOST_TEST(cnt == static_cast<int>(SomeEnum::F));
}

template <typename NoExceptionCallable, typename ExceptionCallable>
class RAII2 final {
  static_assert(std::is_nothrow_invocable_v<ExceptionCallable>);
  static_assert(!std::is_reference_v<NoExceptionCallable>);
  NoExceptionCallable callNoException;
  ExceptionCallable callException;

public:
  RAII2(NoExceptionCallable &&callAlways)
      : callNoException(std::move(callAlways)), callException(callNoException) {
    static_assert(std::is_reference_v<ExceptionCallable>);
  }

  RAII2(NoExceptionCallable &&callNoException,
        ExceptionCallable &&callException)
      : callNoException(std::move(callNoException)),
        callException(std::move(callException)) {}

  RAII2(RAII2 const &) = delete;
  RAII2(RAII2 &&) = delete;
  RAII2 &operator=(RAII2 const &) = delete;
  RAII2 &operator=(RAII2 &&) = delete;

  ~RAII2() noexcept(std::is_nothrow_invocable_v<NoExceptionCallable>) {
    if (!std::uncaught_exceptions())
      callNoException();
    else
      callException();
  }
};
template <typename NoExceptionCallable>
RAII2(NoExceptionCallable &&)
    -> RAII2<std::remove_reference_t<NoExceptionCallable>,
             std::add_lvalue_reference_t<NoExceptionCallable>>;

template <class Policy>
concept CallPolicy = std::is_same_v<Policy, Utility::CallAlways> ||
                     std::is_same_v<Policy, Utility::CallOnException>;
class RAII3 final {
  std::function<void()> callNormally;
  std::function<void()> callOnException;

public:
  RAII3() noexcept {};
  template <std::invocable Callable, CallPolicy Policy = Utility::CallAlways>
  RAII3(Callable &&callable, Policy = Utility::CallAlways{}) {
    static_assert(std::is_nothrow_invocable_v<Callable>);
    if constexpr (std::is_same_v<Policy, Utility::CallAlways>)
      callNormally = std::forward<Callable>(callable);
    else
      callOnException = std::forward<Callable>(callable);
  }

  template <std::invocable Callable, std::invocable CallableOnException>
  RAII3(Callable &&callNormally, CallableOnException &&callOnException)
      : callNormally(std::forward<Callable>(callNormally)),
        callOnException(std::forward<CallableOnException>(callOnException)) {
    static_assert(std::is_nothrow_invocable_v<CallableOnException>);
  }

  RAII3(RAII3 const &) = delete;
  RAII3(RAII3 &&other) noexcept { swap(other); }
  RAII3 &operator=(RAII3 const &) = delete;
  RAII3 &operator=(RAII3 &&other) noexcept {
    swap(other);
    return *this;
  }

  void swap(RAII3 &other) noexcept {
    std::swap(callNormally, other.callNormally);
    std::swap(callOnException, other.callOnException);
  }

  ~RAII3() {
    if (!std::uncaught_exceptions())
      CallNormally();
    else
      CallOnException();
  }

private:
  void CallNormally() {
    if (callNormally)
      callNormally();
  }
  void CallOnException() noexcept {
    if (callOnException)
      callOnException();
    else if (callNormally)
      callNormally();
  }
};

BOOST_AUTO_TEST_CASE(raii_bench_test) {
  constexpr static auto n = 2;
  struct S {
    std::array<int, n> v;
    int hash = 0;
    S() { std::iota(v.begin(), v.end(), 0); }
    int operator()() noexcept {
      auto res = 0;
      std::transform(v.begin(), v.end(), v.begin(),
                     [](int i) { return (i + 1) % n; });
      for (auto i = 0; i != n; ++i)
        res = v[res];
      hash += res;
      return hash;
    }
  };

  auto raii = [s = S{}]() noexcept { Utility::RAII(S{s}); };
  auto raii2 = [s = S{}]() noexcept { RAII2(S{s}); };
  auto raii3 = [s = S{}]() noexcept { RAII3(S{s}); };
  auto s = S{};
  auto ol = std::optional(s);
  auto f = std::function(s);

  auto lt = Utility::Benchmark(s, 1000);
  auto ft = Utility::Benchmark(f, 1000);
  auto olt = Utility::Benchmark(ol.value(), 1000);
  auto rt = Utility::Benchmark(raii, 1000);
  auto rt2 = Utility::Benchmark(raii2, 1000);
  auto rt3 = Utility::Benchmark(raii3, 1000);

  auto x = static_cast<double>(rt.count());
  auto y = static_cast<double>(rt2.count());
  auto overhead = (x - y) / y;
  BOOST_TEST(overhead < 1);

  if (s()) {
    std::cout << "Hash:   " << s() << ol.value().hash << f() << std::endl;
    std::cout << "Lambda test:   " << lt.count() << std::endl;
    std::cout << "Optional Lambda test: " << olt.count() << std::endl;
    std::cout << "Function test: " << ft.count() << std::endl;
    std::cout << "RAII test: " << rt.count() << std::endl;
    std::cout << "RAII2 test: " << rt2.count() << std::endl;
    std::cout << "RAII3 test: " << rt3.count() << std::endl;
  }
}

template <typename T> class SaveRestore2 final {
  static_assert(std::is_nothrow_move_assignable_v<T>);
  static_assert(!std::is_reference_v<T>);
  struct Restore {
    T originalValue;
    T *restoreTo = nullptr;
    void operator()() noexcept { *restoreTo = std::move(originalValue); }
  };
  Utility::RAII<Restore> restore;

public:
  constexpr SaveRestore2() noexcept {};
  explicit SaveRestore2(T &value) noexcept(
      std::is_nothrow_copy_constructible_v<T>)
      : restore{{T{std::as_const(value)}, &value}} {}
  explicit SaveRestore2(T &&value) noexcept(
      std::is_nothrow_move_constructible_v<T>)
      : restore{{std::move(value), &value}} {}
  explicit SaveRestore2(T &&value, T &restoreTo) noexcept(
      std::is_nothrow_move_constructible_v<T>)
      : restore{{std::move(value), &restoreTo}} {}
};

BOOST_AUTO_TEST_CASE(saverestore_bench_test) {
  auto v = 1000;
  auto vcheck = v;
  auto hash = size_t{0};

  auto sr = [&]() {
    {
      auto s = Utility::SaveRestore(v);
      v = 0;
      hash += v;
    }
    hash += v;
  };
  auto sr2 = [&]() {
    {
      auto s = SaveRestore2(v);
      v = 0;
      hash += v;
    }
    hash += v;
  };
  auto srt = Utility::Benchmark(sr, 1000);
  auto srt2 = Utility::Benchmark(sr2, 1000);
  BOOST_TEST(v == vcheck);

  auto x = static_cast<double>(srt.count());
  auto y = static_cast<double>(srt2.count());
  auto overhead = (x - y) / y;
  BOOST_TEST(overhead < 1);

  if (hash == 0) {
    std::cout << "Hash:   " << hash << std::endl;
    std::cout << "SaveRestore test:  " << srt.count() << std::endl;
    std::cout << "SaveRestore2 test: " << srt2.count() << std::endl;
  }
}
