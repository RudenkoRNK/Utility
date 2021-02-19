#pragma once

#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace Utility {

template <typename T, template <typename...> typename Template>
struct _isInstanceOf : std::false_type {};
template <template <typename...> typename Template, typename... Args>
struct _isInstanceOf<Template<Args...>, Template> : std::true_type {};

template <template <typename...> typename Template, typename Instance>
auto constexpr static isInstanceOf = _isInstanceOf<Instance, Template>::value;

template <typename Callable> struct CallableTraits final {
private:
  enum class CallableKind { Function, Method, Lambda };

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

  template <typename Callable_, CallableKind> struct ArgTypes_;
  template <typename Callable_>
  struct ArgTypes_<Callable_, CallableKind::Function> {
    using Types = typename FunctionArgTypes<Callable_>::Types;
    auto constexpr static isCallableConst = true;
  };
  template <typename Callable_>
  struct ArgTypes_<Callable_, CallableKind::Lambda> {
    using Types = typename LambdaOrMethodArgTypes<decltype(
        &Callable_::operator())>::Types;
    auto constexpr static isCallableConst = LambdaOrMethodArgTypes<decltype(
        &Callable_::operator())>::isCallableConst;
  };
  template <typename Callable_>
  struct ArgTypes_<Callable_, CallableKind::Method> {
    using Types = typename LambdaOrMethodArgTypes<Callable_>::Types;
    auto constexpr static isCallableConst =
        LambdaOrMethodArgTypes<Callable_>::isCallableConst;
  };

  template <class Callable_> CallableKind constexpr static GetCallableKind() {
    if (std ::is_function_v<Callable_>)
      return CallableKind::Function;
    if (std::is_class_v<Callable_>)
      return CallableKind::Lambda;
    return CallableKind::Method;
  }

  template <class Callable_, size_t... Indices>
  auto static constexpr GenerateFunctionType(
      std::integer_sequence<size_t, Indices...>)
      -> std::function<typename CallableTraits<Callable_>::ReturnType(
          typename CallableTraits<Callable_>::template ArgType<Indices>...)>;

  using RawCallable =
      typename std::remove_cv_t<std::remove_reference_t<Callable>>;
  using ArgTypes = ArgTypes_<RawCallable, GetCallableKind<RawCallable>()>;
  using Types = typename ArgTypes::Types;

public:
  size_t constexpr static nTypes = std::tuple_size_v<Types>;
  size_t constexpr static nArguments = nTypes - 1;
  template <size_t n> using Type = std::tuple_element_t<n, Types>;
  template <size_t n> using ArgType = std::tuple_element_t<n + 1, Types>;
  using ReturnType = std::tuple_element_t<0, Types>;
  using std_function = decltype(GenerateFunctionType<RawCallable>(
      std::make_index_sequence<nArguments>{}));
  template <size_t n>
  bool constexpr static isLValueReference = std::is_lvalue_reference_v<Type<n>>;
  template <size_t n>
  bool constexpr static isRValueReference = std::is_rvalue_reference_v<Type<n>>;
  template <size_t n>
  bool constexpr static isReference = std::is_reference_v<Type<n>>;
  template <size_t n>
  bool constexpr static isValue = !isReference<n> && !std::is_void_v<Type<n>>;
  template <size_t n>
  bool constexpr static isConst =
      std::is_const_v<std::remove_reference_t<Type<n>>>;
  bool constexpr static isCallableConst = ArgTypes::isCallableConst;

  template <size_t n>
  constexpr static decltype(auto)
  Forward(std::add_lvalue_reference_t<std::remove_reference_t<Type<n>>>
              arg) noexcept {
    if constexpr (isValue<n> || isRValueReference<n>)
      return std::move(arg);
    else
      return arg;
  }
};
} // namespace Utility
