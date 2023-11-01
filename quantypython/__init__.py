from __future__ import annotations

import numpy as np

from fractions import Fraction
from functools import total_ordering, reduce
from numbers import Real
from operator import mul
from typing import Dict, Union, Tuple
from warnings import warn
from weakref import WeakSet

__all__ = ['DimensionError', 'Quantity', 'Unit', 'UnitSystem']


class DimensionError(ValueError):
    """ Error while processing dimensions of quantities. """


@total_ordering
class Quantity:
    """ A dimensional quantity. """

    __slots__ = ['__weakref__', '_value', '_coords']

    _refs = WeakSet()
    _dims = []

    def __init__(self, value: Real, coords: np.ndarray):
        """ This should not be called directly! Use units, numbers and operators to create quantities. """
        self._value = float(value)
        self._coords = coords
        self._refs.add(self)

    def __add__(self, other) -> Quantity:
        if isinstance(other, Quantity):
            if not np.array_equal(self._coords, other._coords):
                raise DimensionError('dimensions don\'t match')
            return Quantity(self._value + other._value, self._coords)
        raise NotImplementedError()

    def __sub__(self, other) -> Quantity:
        if isinstance(other, Quantity):
            if not np.array_equal(self._coords, other._coords):
                raise DimensionError('dimensions don\'t match')
            return Quantity(self._value - other._value, self._coords)
        raise NotImplementedError()

    def __mul__(self, other) -> Union[Quantity, Real]:
        if isinstance(other, Real):
            return Quantity(self._value * other, self._coords)
        if isinstance(other, Quantity):
            coords = self._coords + other._coords
            if not np.any(coords):
                return self._value
            return Quantity(self._value * other._value, coords)
        raise NotImplementedError()

    def __rmul__(self, other) -> Quantity:
        if isinstance(other, Real):
            return Quantity(self._value * other, self._coords)
        raise NotImplementedError()

    def __truediv__(self, other) -> Union[Quantity, Real]:
        if isinstance(other, Real):
            return Quantity(self._value / other, self._coords)
        if isinstance(other, Quantity):
            coords = self._coords - other._coords
            if not np.any(coords):
                return self._value
            return Quantity(self._value / other._value, coords)
        raise NotImplementedError()

    def __rtruediv__(self, other) -> Quantity:
        if isinstance(other, Real):
            return Quantity(other / self._value, -self._coords)
        raise NotImplementedError()

    def __pow__(self, power) -> Union[Quantity, int]:
        if isinstance(power, Real):
            if power == 0:
                return 1
            return Quantity(self._value ** power, self._coords * power)
        raise NotImplementedError()

    def __abs__(self) -> Quantity:
        return Quantity(self._value, self._coords)

    def __floor__(self) -> Quantity:
        return Quantity(np.floor(self._value), self._coords)

    def __ceil__(self) -> Quantity:
        return Quantity(np.floor(self._value), self._coords)

    def __bool__(self) -> bool:
        return bool(self._value)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Quantity) and
            self._value == other._value and
            np.array_equal(self._coords, other._coords)
        )

    def __le__(self, other) -> bool:
        if isinstance(other, Quantity):
            if not np.array_equal(self._coords, other._coords):
                raise DimensionError('dimensions don\'t match')
            return self._value <= other._value
        raise NotImplementedError()

    def __hash__(self):
        return hash(self._value) ^ hash(tuple(self._coords))

    def __repr__(self):
        return f'<Quantity {self.dim_str()}>'

    @property
    def dims(self) -> Dict[str, Fraction]:
        """ Dimensions of the quantity with their powers. """
        return {dim: Fraction(coord) for dim, coord in zip(self._dims, self._coords) if coord}

    def dim_str(self, mul=' ', power_format='^{}') -> str:
        """ Return a string representation of quantity's dimensionality. """
        return mul.join(str(dim) + power_format.format(coord) if coord > 1 else str(dim)
                        for dim, coord in zip(self._dims, self._coords) if coord)

    def as_unit(self, name: str) -> Unit:
        """ Create a new unit from the quantity. """
        if self._value < 0:
            raise ValueError('units can\'t be negative')
        unit = object.__new__(Unit)
        Quantity.__init__(unit, self._value, self._coords)
        unit._name = name
        return unit


class Unit(Quantity):
    """ A named positive quantity which can be used to create systems of units. """

    __slots__ = ['_name']

    def __init__(self, name: str, dim: str):
        """ Define a unit for a new base dimension. """
        self._name = name
        if dim in self._dims:
            raise ValueError(f'a unit for dimension "{dim}" is already defined')
        self._dims.append(dim)
        for quantity in self._refs:
            quantity._coords = np.append(quantity._coords, 0)
        coords = np.zeros(len(self._dims))
        coords[-1] = 1
        super().__init__(1, coords)

    def __str__(self):
        return self._name

    def __repr__(self):
        return f'<Unit "{self._name}": {self.dim_str()}>'


class UnitSystem:
    """ A system of independent units through which quantities are expressed. """

    def __init__(self, *units: Unit):
        """ Create a new system of units. """
        for unit in units:
            if not isinstance(unit, Unit):
                raise TypeError(unit)
        if len(set(unit._name for unit in units)) != len(units):
            warn('Unit system contains units with the same name', RuntimeWarning, 2)
        m = np.array([unit._coords for unit in units])
        if np.linalg.matrix_rank(m) != m.shape[0]:
            raise DimensionError('unit system is redundant')
        self._units = units

    def __repr__(self):
        return '<UnitSystem: ' + ', '.join(f'"{unit}"' for unit in self._units) + '>'

    def __call__(self, quantity: Union[Quantity, Real]) -> Real:
        """ Obtain quantity's value. """
        return self.data(quantity)[0]

    def data(self, quantity: Union[Quantity, Real]) -> Tuple[Real, Dict[Unit, Fraction]]:
        """ Return quantity's value and its units with their powers."""
        if isinstance(quantity, Real):
            return quantity, {}
        if not isinstance(quantity, Quantity):
            raise TypeError(quantity)
        ab = np.array([*(unit._coords for unit in self._units), quantity._coords]).T  # augmented matrix
        ab = ab[ab.any(axis=1)]
        a, b = ab[:, :-1], ab[:, -1]
        if a.shape[0] < a.shape[1]:
            raise DimensionError('unit system is incomplete')
        coords = np.linalg.solve(a, b)
        value = reduce(mul, (unit._value ** coord for unit, coord in zip(self._units, coords)), quantity._value)
        coords = {unit: Fraction(coord).limit_denominator() for unit, coord in zip(self._units, coords)}
        return value, coords

    def str(self, quantity: Union[Quantity, Real], *, value_format='.4g', power_format='^{}', parts_format='{} ({})',
            fraction_format='{}/{}', mul=' * ', div=' / ', denominator_div=True) -> str:
        """ Make a string representation of a quantity. """
        mul1, mul2 = (mul, div) if denominator_div else (mul, mul)
        value, coords = self.data(quantity)
        format_unit_coord = lambda unit, coord: str(unit) if coord == 1 else str(unit) + power_format.format(
            coord if coord.denominator == 1 else fraction_format.format(coord.numerator, coord.denominator))
        value = None if value_format is None else f'{value:{value_format}}'
        if div is None:
            terms = mul1.join(format_unit_coord(unit, coord) for unit, coord in coords.items() if coord)
            if parts_format is None:
                if value is None:
                    return terms
                return f'{value}{mul1}{terms}'
            return parts_format.format(value, terms)
        numerator = mul1.join(format_unit_coord(unit, coord) for unit, coord in coords.items() if coord > 0)
        denominator = mul2.join(format_unit_coord(unit, -coord) for unit, coord in coords.items() if coord < 0)
        if denominator:
            if numerator:
                if value is None:
                    return f'{numerator}{div}{denominator}'
                if parts_format is None:
                    return f'{value}{mul1}{numerator}{div}{denominator}'
                return parts_format.format(value, f'{numerator}{div}{denominator}')
            if value is None:
                if parts_format is None:
                    return denominator
                return f'1{div}{denominator}'
            if parts_format is None:
                return f'{value}{div}{denominator}'
            return parts_format.format(value, f'1{div}{denominator}')
        if value is None:
            return numerator
        if parts_format is None:
            return f'{value}{mul1}{numerator}'
        return parts_format.format(value, f'{numerator}')
