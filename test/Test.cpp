#define BOOST_TEST_MODULE Test

#include "Utility/Math.hpp"
#include "Utility/Misc.hpp"
#include "Utility/TypeTraits.hpp"
#include <boost/test/included/unit_test.hpp>
#include <functional>
#include <random>

BOOST_AUTO_TEST_CASE(arg_traits_test) {
  auto lambda1 = [](std::string const &) { return 0; };
  using T = Utility::ArgumentTraits<decltype(lambda1)>::Type<1>;
  // NOLINTNEXTLINE
  auto lambda2 = [&](T t) { return lambda1(t); };

  BOOST_TEST(Utility::ArgumentTraits<decltype(lambda1)>::isConst<1>);
  BOOST_TEST(Utility::ArgumentTraits<decltype(lambda1)>::isLValueReference<1>);
  BOOST_TEST(Utility::ArgumentTraits<decltype(lambda2)>::isConst<1>);
  BOOST_TEST(Utility::ArgumentTraits<decltype(lambda2)>::isLValueReference<1>);
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

  BOOST_TEST(!Utility::ArgumentTraits<decltype(lambda1)>::isCallableConst);
  BOOST_TEST(!Utility::ArgumentTraits<decltype(lambda2)>::isCallableConst);
  BOOST_TEST(Utility::ArgumentTraits<decltype(lambda3)>::isCallableConst);
  BOOST_TEST(Utility::ArgumentTraits<decltype(lambda4)>::isCallableConst);
}

BOOST_AUTO_TEST_CASE(perm_test) {
  auto perm = std::vector<size_t>{5, 2, 3, 0, 1, 4};
  auto v = std::vector<size_t>(perm.size());
  std::iota(v.begin(), v.end(), 0);
  Utility::Permute(v, perm);
  BOOST_TEST(v == perm);
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
