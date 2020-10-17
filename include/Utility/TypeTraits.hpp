#pragma once

#include <functional>
#include <tuple>
#include <type_traits>

namespace Utility {

class TypeTraits final {
private:
  template <typename T, template <typename...> typename Template>
  struct isInstanceOf_ : std::false_type {};
  template <template <typename...> typename Template, typename... Args>
  struct isInstanceOf_<Template<Args...>, Template> : std::true_type {};

public:
  template <typename Instance, template <typename...> typename Template>
  auto constexpr static isInstanceOf =
      isInstanceOf_<std::remove_reference_t<Instance>, Template>::value;
};

template <typename Callable> struct ArgumentTraits final {
private:
  enum class CallableType { Function, Method, Lambda };

  template <typename Callable_> struct FunctionArgTypes;
  template <typename Callable_, typename... Args>
  struct FunctionArgTypes<Callable_(Args...)> {
    using Types = typename std::tuple<Callable_, Args...>;
  };
  template <typename CallableAddress> struct LambdaOrMethodArgTypes;
  template <typename CallableAddress, typename Result, typename... Args>
  struct LambdaOrMethodArgTypes<Result (CallableAddress::*)(Args...) const> {
    using Types = typename std::tuple<Result, Args...>;
    auto constexpr static isCallableConst = true;
  };
  template <typename CallableAddress, typename Result, typename... Args>
  struct LambdaOrMethodArgTypes<Result (CallableAddress::*)(Args...)> {
    using Types = typename std::tuple<Result, Args...>;
    auto constexpr static isCallableConst = false;
  };

  template <typename Callable_, CallableType> struct ArgTypes;
  template <typename Callable_>
  struct ArgTypes<Callable_, CallableType::Function> {
    using Types = typename FunctionArgTypes<Callable_>::Types;
    auto constexpr static isCallableConst = false;
  };
  template <typename Callable_>
  struct ArgTypes<Callable_, CallableType::Lambda> {
    using Types = typename LambdaOrMethodArgTypes<decltype(
        &Callable_::operator())>::Types;
    auto constexpr static isCallableConst = LambdaOrMethodArgTypes<decltype(
        &Callable_::operator())>::isCallableConst;
  };
  template <typename Callable_>
  struct ArgTypes<Callable_, CallableType::Method> {
    using Types = typename LambdaOrMethodArgTypes<Callable_>::Types;
    auto constexpr static isCallableConst =
        LambdaOrMethodArgTypes<Callable_>::isCallableConst;
  };

  auto constexpr static ThisCallableType = std::is_function_v<Callable>
                                               ? CallableType::Function
                                               : std::is_class_v<Callable>
                                                     ? CallableType::Lambda
                                                     : CallableType::Method;

  using ArgTypes_ = typename ArgTypes<Callable, ThisCallableType>;
  using Types = typename ArgTypes_::Types;

public:
  size_t constexpr static nArguments = std::tuple_size_v<Types> - 1;
  template <size_t n> using Type = std::tuple_element_t<n, Types>;
  template <size_t n>
  bool constexpr static isLValueReference = std::is_lvalue_reference_v<Type<n>>;
  template <size_t n>
  bool constexpr static isRValueReference = std::is_rvalue_reference_v<Type<n>>;
  template <size_t n>
  bool constexpr static isReference = std::is_reference_v<Type<n>>;
  template <size_t n> bool constexpr static isValue = !isReference<n>;
  template <size_t n>
  bool constexpr static isConst =
      std::is_const_v<std::remove_reference_t<Type<n>>>;
  bool constexpr static isCallableConst = ArgTypes_::isCallableConst;
};
} // namespace Utility
