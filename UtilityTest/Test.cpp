#define BOOST_TEST_MODULE Test

#include "Utility/Math.hpp"
#include "Utility/Misc.hpp"
#include "Utility/TypeTraits.hpp"
#include <boost/test/included/unit_test.hpp>
#include <complex>
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
  BOOST_TEST(
      (Utility::TypeTraits::isInstanceOf<std::vector, std::vector<int>>));
  BOOST_TEST((!Utility::TypeTraits::isInstanceOf<std::vector,
                                                 std::unordered_set<int>>));
  BOOST_TEST((!Utility::TypeTraits::isInstanceOf<std::vector, int>));
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

  auto i = 123456;
  auto checki = i;
  {
    auto save = Utility::SaveRestore(i);
    i = 100;
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

BOOST_AUTO_TEST_CASE(unordered_hash_test) {
  auto a = std::unordered_set<int>{};
  auto b = a;
  auto c = a;
  auto x = std::unordered_map<int, double>{};
  auto y = x;
  auto z = y;
  a.insert({0, 1, 2, 3, 4, 5});
  b.insert({3, 2, 0, 4, 5, 1});
  c.insert({1, 2, 3, 4, 5});

  x.insert({{0, 0.0}, {1, 1.0}, {2, 2.0}, {3, 3.0}});
  y.insert({{1, 1.0}, {0, 0.0}, {3, 3.0}, {2, 2.0}});
  z.insert({{0, 0.0}, {1, 1.0}, {2, 3.0}, {3, 3.0}});
  BOOST_TEST(Utility::UnorderedHash(a) == Utility::UnorderedHash(b));
  BOOST_TEST(Utility::UnorderedHash(a) != Utility::UnorderedHash(c));
  BOOST_TEST(Utility::UnorderedHash(x) == Utility::UnorderedHash(y));
  BOOST_TEST(Utility::UnorderedHash(x) != Utility::UnorderedHash(z));
}

BOOST_AUTO_TEST_CASE(enumerate_test) {
  auto a = std::unordered_set<int>{4, 3, 2, 1};
  auto b = std::vector<int>{4, 3, 2, 1};
  auto am = Utility::Enumerate(a);
  auto bm = Utility::Enumerate(b);
  auto acheck = std::unordered_set<size_t>{};
  auto bcheck = std::unordered_set<size_t>{};
  for (auto const &[e, i] : am)
    acheck.insert(i);
  for (auto const &[e, i] : bm)
    bcheck.insert(i);

  BOOST_TEST(acheck.size() == a.size());
  BOOST_TEST(bcheck.size() == b.size());

  BOOST_TEST(*std::min_element(acheck.begin(), acheck.end()) == 0);
  BOOST_TEST(*std::max_element(acheck.begin(), acheck.end()) ==
             acheck.size() - 1);
  BOOST_TEST(*std::min_element(bcheck.begin(), bcheck.end()) == 0);
  BOOST_TEST(*std::max_element(bcheck.begin(), bcheck.end()) ==
             bcheck.size() - 1);
}
struct AAA {
  int x;
  int y;
  AAA() = delete;
  AAA(int x, int y) : x(x), y(y){};
};
static bool operator==(AAA const &lhs, AAA const &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y;
}

BOOST_AUTO_TEST_CASE(id_wrapper_test) {
  using IdInt = Utility::IdWrapper<int>;
  auto x1 = IdInt(1, 0);
  auto x2 = IdInt(3, 1);
  auto x3 = IdInt(4, 2);
  auto y1 = Utility::IdWrapper<AAA>({1, 1}, 0);
  auto y2 = Utility::IdWrapper<AAA>({1, 2}, 1);
  auto y3 = Utility::IdWrapper<AAA>({1, 3}, 2);

  BOOST_TEST(x1.Id() == 0);
  BOOST_TEST(x2.Id() == 1);
  BOOST_TEST(x3.Id() == 2);
  BOOST_TEST(y1.Id() == 0);
  BOOST_TEST(y2.Id() == 1);
  BOOST_TEST(y3.Id() == 2);
  BOOST_TEST(x2 == 3);
  BOOST_TEST(x3 == 4);
  assert(y2 == AAA(1, 2));
  assert(y3 == AAA(1, 3));

  auto xv = std::vector<IdInt>{x1};
  auto x4 = x1;

  BOOST_TEST(x1.Id() == xv[0].Id());
  BOOST_TEST(x4.Id() == x1.Id());
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
}
