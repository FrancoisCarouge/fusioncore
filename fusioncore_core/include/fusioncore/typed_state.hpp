#pragma once

// FusionCore's adapter of the typed linear algebra to mp-units elements, and
// the quantities of the state's elements: the measurement functions build
// their expected measurements as typed vectors, each component checked
// against its unit, and hand the filter their numerical values.

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <Eigen/Dense>

#include "fcarouge/sparse_linear_algebra.hpp"
#include "fcarouge/typed_linear_algebra.hpp"
#include "mp-units/framework/quantity.h"
#include "mp-units/systems/si.h"

namespace fcarouge {
// A quantity element is stored as its numerical value in its own unit, and a
// stored value is read back as a quantity of that unit.
template <typename To, mp_units::Quantity From>
struct element_caster<To, From> {
  [[nodiscard]] static constexpr auto operator()(From value) -> To {
    static_assert(std::same_as<typename From::rep, std::remove_cvref_t<To>>);
    return value.numerical_value_in(From::unit);
  }
};

template <mp_units::Quantity To, typename From>
struct element_caster<To, From> {
  [[nodiscard]] static constexpr auto operator()(From value) -> To {
    static_assert(std::same_as<typename To::rep, std::remove_cvref_t<From>>);
    return value * To::reference;
  }
};
}  // namespace fcarouge

// A typed matrix of quantities is never a quantity's representation: without
// this, mp-units' `quantity * value` tests the matrix as a representation,
// `matrix * mag<1>`, which fails to compile inside the typed linear algebra
// instead of declining. Every typed matrix here holds quantities.
template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
inline constexpr bool mp_units::disable_representation<
    fcarouge::typed_matrix<Matrix, RowIndexes, ColumnIndexes>>{true};

namespace fusioncore {
namespace typed {

template <auto Reference>
using Quantity = mp_units::quantity<Reference, double>;

inline constexpr auto metre = mp_units::si::metre;
inline constexpr auto second = mp_units::si::second;
inline constexpr auto radian = mp_units::si::radian;

using Length = Quantity<metre>;
using Ratio = Quantity<mp_units::one>;
using Angle = Quantity<radian>;
using Speed = Quantity<metre / second>;
using Acceleration = Quantity<metre / (second * second)>;
// An angular rate, the angle a ratio: mp-units keeps the radian's kind apart,
// and a lever arm's m rad/s does not add to m/s.
using AngularSpeed = Quantity<mp_units::one / second>;

// A column vector of the given element types.
template <typename... Types>
using Vector = fcarouge::typed_column_vector<
    fcarouge::sparse_matrix<double, fcarouge::sparse::dense<sizeof...(Types), 1>>, Types...>;

// The state element at `index` as a quantity of type `Type`, its documented
// unit (state.hpp).
template <typename Type, typename Untyped>
[[nodiscard]] Type element(const Untyped& x, int index) {
  return x[index] * Type::reference;
}

// The numerical values of the typed column vector `value`, each in its unit.
template <typename Typed>
[[nodiscard]] Eigen::Matrix<double, Typed::rows, 1> untyped(const Typed& value) {
  Eigen::Matrix<double, Typed::rows, 1> result;
  [&]<std::size_t... Index>(std::index_sequence<Index...>) {
    ((result[Index] = value.template at<Index>().numerical_value_in(
          fcarouge::tla::element_at<Typed, Index, 0>::unit)),
     ...);
  }(std::make_index_sequence<Typed::rows>{});
  return result;
}

}  // namespace typed
}  // namespace fusioncore
