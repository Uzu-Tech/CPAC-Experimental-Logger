from typing import Any, Dict, Iterable, List, Tuple
from equations import format_to_sig_figs

import pandas as pd

def calculate_num_readings(
    start_value: float, end_value: float, interval: float
) -> int:
    """Calculate the number of readings based on start, end, and interval."""
    return int((end_value - start_value) / interval) + 1


def generate_independent_variable_data(
    start_value: float, num_readings: int, interval: float
) -> Tuple[float]:
    """Generate data for the independent variable."""
    return tuple(
        round(start_value + reading_num * interval, 4)
        for reading_num in range(num_readings)
    )


def generate_empty_dependent_data(
    num_repeats: int, num_readings: int
) -> Tuple[Tuple[float]]:
    """Generate empty dependent data."""
    if num_repeats > 1:
        return tuple((0.0,) * num_readings for _ in range(num_repeats))
    return (0.0,) * num_readings


def get_initial_table_from_details(
    independent_details: Dict[str, Any],
    dependent_details: Dict[str, Any],
) -> Dict[str, Any]:
    data = {}

    start_value = independent_details["Starting Value"]
    interval = independent_details["Interval Size"]
    num_readings = independent_details["No. Readings"]
    num_repeats = dependent_details["No. Repeats"]
    independent_var_name = independent_details["Name"]
    independent_unit = independent_details["Unit"]
    dependent_unit = dependent_details["Unit"]
    dependent_var_name = dependent_details["Name"]

    independent_data = generate_independent_variable_data(
        start_value, num_readings, interval
    )
    empty_data_columns = generate_empty_dependent_data(num_repeats, num_readings)

    data[independent_var_name + f" ({independent_unit})"] = independent_data
    if num_repeats > 1:
        for i, column in enumerate(empty_data_columns):
            data[dependent_var_name + f" ({dependent_unit})" + f" {str(i + 1)}"] = (
                column
            )
    else:
        data[dependent_var_name + f" ({dependent_unit})"] = empty_data_columns

    return data


def create_dataframe(
    data: Iterable[Iterable] | Dict[str, Any], headers_row=False
) -> pd.DataFrame:
    """Create a DataFrame from a 2D array of data or a dictionary."""
    if headers_row:
        data = list(data)
        headers = tuple(data[0])
        data = data[1:]
        return pd.DataFrame(data, columns=headers)

    return pd.DataFrame(data)


def convert_to_array(data: pd.DataFrame) -> List[List]:
    """Convert a Dataframe to a 2D array of data"""
    # Add headers first
    arr = [
        list(data),
    ]
    # Add the rest of data
    arr.extend(list(map(format_to_sig_figs, row)) for row in data.to_numpy())
    return arr
