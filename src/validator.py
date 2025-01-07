import re
from typing import Any, Callable, Dict, Tuple
from fractions import Fraction

import equations
from uncertainties import ufloat


def validate_formula(expression: str):
    # Define the regex pattern for a valid math formula
    # This pattern enforces that operators must be between valid operands (numbers or variables)
    pattern = re.compile(
        r"^\s*"  # Leading whitespace
        r"\-?\s*(\d+|[a-zA-Z]|\(\s*\d+|\(\s*[a-zA-Z])"  # First operand: digit, variable, or expression in parentheses
        r"(\s*[+\-*/^]\s*\-?\s*"  # Operator with optional surrounding whitespace
        r"(\d+|[a-zA-Z]|\(*\s*\d+\s*\)*|\(*\s*[a-zA-Z]\s*\)*)\s*)*$"  # Subsequent operands and operators
    )

    # Match the pattern against the input expression
    match = pattern.fullmatch(expression.strip())
    if not match:
        return (
            f'Gradient Formula: "{expression}" is not a valid formula.'
            + ' Useable operators include (*, +, -, /, ^), also "4x" for example must'
            + ' be written as "4 * x" instead.'
        )

    # Check all brackets are closed
    stack = []
    bracket_pairs = {"(": ")", "[": "]", "{": "}"}
    bracket_err_message = (
        f'Gradient Formula: "{expression}" must have closed bracket pairs'
    )

    for char in expression:
        if char in bracket_pairs:  # If it's an opening bracket
            stack.append(char)
        elif char in bracket_pairs.values():  # If it's a closing bracket
            if not stack or bracket_pairs[stack.pop()] != char:
                return bracket_err_message
    # If open brackets are still left over
    if stack:
        return bracket_err_message

    # Check if there's at least one alphabetic letter
    if not any(c.isalpha() for c in expression):
        return f'Gradient Formula: "{expression}" must have at least one symbol'

    return None


def validate_dict(
    data: Dict,
    dtypes: Tuple[type],
    error_function: Callable,
    default_values: Dict[str, Any] = None,
    additional_checks: Dict[str, Callable] = None,
):
    new_dict = data.copy()
    for key, dtype in zip(data, dtypes):
        value = data[key]
        # Check if required fields are not entered
        if value == "":
            if default_values is None or key not in default_values:
                error_function(f"{key}: {key} is a required field")
                return None
            else:
                value = default_values[key]
                new_dict[key] = value
        # Check for incorrect typing
        try:
            value = dtype(value)
            new_dict[key] = value
        except ValueError:
            error_function(f'{key}: "{value}" is an incorrect type')
            return None
        # Check for any custom conditions
        if (
            additional_checks is not None
            and (additional_check := additional_checks.get(key)) is not None
        ):
            error = additional_check(value)
            if error is not None:
                error_function(error)
                return None
    return new_dict


def is_readings_input_correct(readings: str) -> bool:
    pattern = re.compile(
        r"(?:\s*(\d+)"  # Number
        r"(?:\s+|\s*\,\s*)*)*"  # Comma or white space
    )

    if not pattern.fullmatch(readings.strip()):
        return (
            f'Remove Readings: "{readings}" is invalid, expected numbers separated'
            + "by commas or whitespace (e.g. 1 2 4 or 1, 2, 4)"
        )

    return None


def get_readings_to_delete(readings: str) -> Tuple[int, ...]:
    # Split the input by commas and whitespace, filter out empty strings, and convert to integers
    readings_list = re.split(r"[,\s]+", readings.strip())
    return tuple(int(reading) for reading in readings_list if reading)


def validate_unit(unit):
    if not unit.isalpha():
        return "Unit: Unit should only contain letters"
    if len(unit) != 1 and len(unit) != 2:
        return "Unit: Unit can only be 2 or 1 letter long (e.g. cm or N))"
    if len(unit) > 1 and unit[0] not in equations.prefix_lookup:
        return f"Unit: {unit[0]} is not a valid prefix, examples include {tuple(equations.prefix_lookup)}"
    return None


def is_letters_and_spaces(string, field):
    if not all(c.isalpha() or c.isspace() for c in string):
        return f"{field}: {string} is not valid, name must be fully alphabetic"
    return None


def get_ufloat(input_str, variable, error_func):
    # Define a regular expression pattern to extract the value and uncertainty
    pattern = re.compile(
        r"(\-?\d*\.?\d+(?:[eE]\-?\d+)?)\s*(?:\+/-\s*(\d*\.?\d+(?:[eE]\-?\d+)?))?"
    )

    # Search for the pattern in the input string
    match = re.search(pattern, input_str.strip())
    full_match = pattern.fullmatch(input_str.strip())

    if full_match:
        # Extract the value from the match groups
        value = float(match.group(1))

        # Extract the uncertainty if it exists
        if match.group(2):
            uncertainty = float(match.group(2))
            # Create and return the ufloat object
            return ufloat(value, uncertainty)
        else:
            # If no uncertainty is provided, assume it has no uncertainty
            return ufloat(value, 0)
    else:
        # Handle the error if the input string does not match the expected pattern
        error_func(
            f"{variable}: Expected format 'value +/- uncertainty' or 'value'."
            + " Value must also be positive"
        )
        return None

def exponent(x: str):
    pattern = re.compile(r'\-?\d+|\-?\d*\.\d+|\-?\d+/[1-9]\d*')
    match = pattern.fullmatch(x)
    if not match:
        raise ValueError

    if "/" in match.group(0):
        return Fraction(match.group(0)) 
    else:
        return float(match.group(0))
