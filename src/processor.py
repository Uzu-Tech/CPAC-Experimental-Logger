from math import floor, log10
from typing import Any, Callable, Iterable, Tuple

import numpy as np
import pandas as pd


def dataframe_to_dict(data: pd.DataFrame):
    "Converts a dataframe to a dictionary"
    return {header: data[header] for header in tuple(data)}


def get_dependant_data(data: pd.DataFrame) -> np.ndarray:
    """Gets the dependent data by deleting the first column of independent data."""
    return np.delete(data.to_numpy(), 0, 1)


def calculate_means(data: pd.DataFrame, decimal_places: int = 2) -> np.ndarray:
    """Calculate the means of dependent data."""
    dependant_data = get_dependant_data(data)
    return np.round(np.mean(dependant_data, axis=1), decimals=decimal_places)


def calculate_absolute_uncertainties(data: pd.DataFrame) -> np.ndarray:
    """Calculate the absolute uncertainties of dependent data."""
    dependant_data = get_dependant_data(data)
    return (np.max(dependant_data, axis=1) - np.min(dependant_data, axis=1)) / 2


def calculate_percentage_uncertainties(
    values: np.ndarray, abs_uncertainties: np.ndarray, decimal_places: int = 1
) -> np.ndarray:
    """Calculate the percentage uncertainties."""
    return np.round(
        np.array(
            [
                uncertainty / value * 100 if value != 0 else 0
                for uncertainty, value in zip(abs_uncertainties, values)
            ]
        ),
        decimals=decimal_places,
    )


def add_column(data: pd.DataFrame, column: Iterable, column_name: str) -> pd.DataFrame:
    """Add a new column to the DataFrame."""
    data.insert(len(data.columns), column_name, column)
    return data


def get_graph_data(
    data: pd.DataFrame, independent_header: str, dependent_header: str
) -> pd.DataFrame:
    """Get data for graphing."""
    graph_data = {}

    graph_data[independent_header] = data[independent_header]
    graph_data[dependent_header] = data[dependent_header]

    return pd.DataFrame(graph_data)


def modify_data_column(
    data: pd.DataFrame, column_header: str, operation: Callable
) -> pd.DataFrame:
    """Modify a specific column in the DataFrame according to the given operation."""
    data_dict = dataframe_to_dict(data)
    data_dict[column_header] = tuple(operation(value) for value in data[column_header])
    return pd.DataFrame(data_dict)


def modify_multiple_columns(
    data: pd.DataFrame, headers: Tuple[str], operations: Tuple[Callable]
) -> pd.DataFrame:
    """Modifies a column in the DataFrame according to the given operations."""
    data_dict = dataframe_to_dict(data)
    for header, operation in zip(headers, operations):
        data_dict[header] = tuple(operation(value) for value in data[header])
    return pd.DataFrame(data_dict)


def get_data_frame_column(data: pd.DataFrame, header: str) -> Tuple:
    """Gets a new data frame with the specified columns."""
    return data[header]


def get_data_frame_columns(data: pd.DataFrame, headers: Tuple[str]):
    """Gets a new data frame with the specified columns."""
    return pd.DataFrame({header: data[header] for header in headers})


def combine_two_columns(
    data_frame: pd.DataFrame, header1: str, header2: str, operation: Callable
) -> Tuple[Any]:
    """Combines two columns in a dataframe based on a specified operation"""
    column1 = data_frame[header1]
    column2 = data_frame[header2]
    return tuple(operation(value1, value2) for value1, value2 in zip(column1, column2))


def get_x_y(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Get X and Y values from the DataFrame."""
    data = data.to_numpy()
    x = data[:, 0]
    y = data[:, 1]
    return x.copy(), y.copy()


def calculate_max_gradient_points(
    x: Iterable,
    y: Iterable,
    errors_x: np.ndarray = None,
    errors_y: np.ndarray = None,
    is_origin_intercept: bool=False
) -> Tuple[np.ndarray, np.ndarray]:
    x, y = np.array(x), np.array(y)

    # If it must pass through origin...
    # take the top left corner of every error box for the highest gradient
    if is_origin_intercept:
        x -= errors_x
        y += errors_y
        return x, y

    # Check if the gradient is negative by checking if one is going down while the other
    # goes up, if only one is going down then x or y difference will be negative while
    # the other positive so their product is negative
    is_gradient_negative = ((y[0] - y[1]) * (x[0] - x[1])) < 0
    # Get the divisors
    divider_1 = len(x) // 2
    # Split points in half if even number
    divider_2 = divider_1 if len(x) % 2 == 0 else divider_1 + 1
    x1, x2 = x[:divider_1], x[divider_2:]
    y1, y2 = y[:divider_1], y[divider_2:]

    if errors_x is not None:
        # Take the biggest values from the first x values
        x1 += errors_x[:divider_1]
        # Take the smallest values from the other x values
        x2 -= errors_x[divider_2:]

    if errors_y is not None:
        if not is_gradient_negative:
            # Take the smallest values from the first y values
            y1 -= errors_y[:divider_1]
            # Take the biggest values from the other y values
            y2 += errors_y[divider_2:]
        else:
            # Take the biggest values from the first y values
            y1 += errors_y[:divider_1]
            # Take the smallest values from the other y values
            y2 -= errors_y[divider_2:]

    if len(x) % 2 == 0:
        x, y = np.concatenate((x1, x2)), np.concatenate((y1, y2))
    else:
        median_x, median_y = np.median(x), np.median(y)
        # Add the middle to the list
        x = (np.concatenate((x1, (median_x,), x2)),)
        y = np.concatenate((y1, (median_y,), y2))

    return x, y


def calculate_min_gradient_points(
    x: Iterable,
    y: Iterable,
    errors_x: np.ndarray = None,
    errors_y: np.ndarray = None,
    is_origin_intercept: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    x, y = np.array(x), np.array(y)

    # If it must pass through origin...
    # take the bottom right corner of every error box for the highest gradient
    if is_origin_intercept:
        x += errors_x
        y -= errors_y
        return x, y

    # Check if the gradient is negative by checking if one is going down while the other
    # goes up, if only one is going down then x or y difference will be negative while
    # the other positive so their product is negative
    is_gradient_negative = ((y[0] - y[1]) * (x[0] - x[1])) < 0
    # Get the divisors
    divider_1 = len(x) // 2
    # Split points in half if even number
    divider_2 = divider_1 if len(x) % 2 == 0 else divider_1 + 1
    x1, x2 = x[:divider_1], x[divider_2:]
    y1, y2 = y[:divider_1], y[divider_2:]

    if errors_x is not None:
        # Take the smallest values from the first x values
        x1 -= errors_x[:divider_1]
        # Take the biggest values from the other x values
        x2 += errors_x[divider_2:]

    if errors_y is not None:
        if not is_gradient_negative:
            # Take the biggest values from the first y values
            y1 += errors_y[:divider_1]
            # Take the smallest values from the other y values
            y2 -= errors_y[divider_2:]
        else:
            # Take the smallest values from the first y values
            y1 -= errors_y[:divider_1]
            # Take the biggest values from the other y values
            y2 += errors_y[divider_2:]

    if len(x) % 2 == 0:
        x, y = np.concatenate((x1, x2)), np.concatenate((y1, y2))
    else:
        median_x, median_y = np.median(x), np.median(y)
        # Add the middle to the list
        x = (np.concatenate((x1, (median_x,), x2)),)
        y = np.concatenate((y1, (median_y,), y2))

    return x, y


def sci_notation(x: float, precision: int):
    """
    Rounds a number to number of significant figures
    Parameters:
    - x - the number to be rounded
    - precision (integer) - the number of significant figures
    Returns:
    - float
    """

    x = float(x)
    precision = int(precision)

    if abs(x) >= 1000 or abs(x) < 0.01:
        return np.format_float_scientific(
            x, precision=precision - 1, sign=False, exp_digits=1
        )

    return np.round(x, -int(floor(log10(abs(x)))) + (precision - 1))
