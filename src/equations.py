from math import floor, log10
from re import findall
from typing import Dict

import sympy as sy
from sympy.solvers.solveset import solveset_real
from uncertainties import ufloat
from uncertainties import Variable

prefix_lookup = {
    "y": 1e-24,  # yocto
    "z": 1e-21,  # zepto
    "a": 1e-18,  # atto
    "f": 1e-15,  # femto
    "p": 1e-12,  # pico
    "n": 1e-9,  # nano
    "u": 1e-6,  # micro
    "m": 1e-3,  # mili
    "c": 1e-2,  # centi
    "d": 1e-1,  # deci
    "": 1,  # none
    "k": 1e3,  # kilo
    "M": 1e6,  # mega
    "G": 1e9,  # giga
    "T": 1e12,  # tera
    "P": 1e15,  # peta
    "E": 1e18,  # exa
    "Z": 1e21,  # zetta
    "Y": 1e24,  # yotta
}


def get_all_constants(equation: str, var_to_solve: str):
    return tuple(
        symbol for symbol in findall("[a-zA-Z]", equation) if symbol != var_to_solve
    )

def get_expression(formula, var_to_solve):
    # Define x as unknown variable
    x = sy.Symbol(var_to_solve)
    symbols = {var_to_solve: x}
    symbols.update(
        {
            letter: sy.Symbol(letter, real=True, positive=True)
            for letter in findall("[a-zA-Z]", formula)
            if letter != var_to_solve
        }
    )

    # Create equation
    return sy.parse_expr(formula, symbols, transformations="all")

def solve_equation(
    formula: str, var_to_solve: str, gradient_value: Variable, constants: Dict = None
):
    # Define x as unknown variable
    x = sy.Symbol(var_to_solve)
    symbols = {var_to_solve: x}
    symbols.update(
        {
            letter: sy.Symbol(letter, real=True, positive=True)
            for letter in findall("[a-zA-Z]", formula)
            if letter != var_to_solve
        }
    )

    # Create equation
    expr = sy.parse_expr(formula, symbols, transformations="all")

    # Define gradient symbol
    gradient = sy.Symbol("grad", real=True, positive=True)
    # Set the equation equal to gradient and solve
    eq = (
        sy.Eq(expr, gradient)
        if gradient_value.nominal_value > 0
        else sy.Eq(expr, -gradient)
    )
    solutions = solveset_real(eq, x)

    # Add the gradient to constants
    constants_values = {gradient: abs(gradient_value.nominal_value)}
    constants_uncertainties = {gradient: gradient_value.std_dev}
    # Separate values from uncertainties
    if constants:
        for constant_str, symbol in symbols.items():
            if symbol != x:
                constants_values[symbol] = constants[constant_str].nominal_value
                constants_uncertainties[symbol] = constants[constant_str].std_dev

    for solution in solutions:
        value = solution.subs(constants_values)
        uncertainty = propagate_error(
            solution, constants_values, constants_uncertainties
        )
        yield ufloat(value, uncertainty)


def propagate_error(
    solution, values: Dict[sy.Symbol, float], uncertainties: Dict[sy.Symbol, float]
):
    """
    Propagate errors through a given equation.

    Parameters:
    - equation: The equation as a sympy expression.
    - variables: List of sympy symbols corresponding to the variables in the equation.
    - uncertainties: List of uncertainties corresponding to each variable.

    Returns:
    - The propagated uncertainty.
    """

    # Compute partial derivatives with respect to each variable
    partial_derivatives = {value: sy.diff(solution, value) for value in values}

    # Compute the magnitude of the gradient
    return sy.sqrt(
        sum(
            (partial_derivatives[value].subs(values) * uncertainties[value]) ** 2
            for value in values
        )
    )


def format_to_sig_figs(x, precision=3) -> str:
    if x == 0:
        return "0"

    x = float(x)
    precision = int(precision)

    # Calculate the order of magnitude
    exponent = int(floor(log10(abs(x))))
    
    # Determine if scientific notation is needed
    is_sci_notation = abs(x) >= 10_000 or abs(x) < 0.001

    # Normalize x to be within [1, 10)
    normalized_x = x / (10**exponent)

    # Round the normalized number to the desired precision
    rounded_num = round(normalized_x, precision - 1)
    
    # Format the output as scientific notation or regular number
    if is_sci_notation:
        # Convert rounded_num to integer if it's effectively an integer
        if int(rounded_num) == rounded_num:
            rounded_num = int(rounded_num)
        return f"{rounded_num}e{exponent}"
    else:
        rounded_num = round(x, -exponent + precision - 1)
        # Convert rounded_num to integer if it's effectively an integer
        if int(rounded_num) == rounded_num:
            rounded_num = int(rounded_num)
        return str(rounded_num)
    

def simplify_float(x: float):
    return sy.nsimplify(x)