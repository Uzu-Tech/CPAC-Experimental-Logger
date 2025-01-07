import tkinter as tk
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Generator, Iterable, List, Tuple
import ctypes

import customtkinter as ctk
import equations
import generator
import gui
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import plotter
import processor
import validator
from uncertainties import ufloat

State = Enum("State", ["No_Table", "Editing", "Finished"])


def create_labels_and_entries(
    frame: ctk.CTkFrame,
    names: Iterable[str],
    sizes: Iterable[float],
    font: Tuple[str, float],
    placeholder_texts: Dict[str, str],
    placeholder_color: str,
) -> Tuple[Tuple[ctk.CTkLabel], Tuple[ctk.CTkEntry], Dict[str, ctk.CTkEntry]]:
    """
    Create labels and entries within a given frame, configure them with specified properties,
    and return them along with a dictionary of variables.

    Parameters
    ----------
    frame : ctk.CTkFrame
        The frame where the labels and entries will be created.
    names : Iterable[str]
        The names for the labels and entries. Each name corresponds to a placeholder text and variable.
    sizes : Iterable[float]
        The relative width sizes for the entries.
    font : Tuple[str, float]
        A tuple specifying the font family and size for the labels and entries.
    placeholder_texts : Dict[str, str]
        A dictionary mapping names to placeholder texts for the entries.
    placeholder_color : str
        The color of the placeholder text.

    Returns
    -------
    Tuple[Tuple[ctk.CTkLabel], Tuple[ctk.CTkEntry], Dict[str, ctk.CTkEntry]]
        A tuple containing:
        - A tuple of created labels.
        - A tuple of created entries.
        - A dictionary mapping names to the created entry variables.
    """
    labels, entries = [], []
    variables = {}

    # Iterate over names and sizes to create labels and entries
    for name, size in zip(names, sizes):
        # Retrieve placeholder text if available; otherwise use an empty string
        placeholder_text = placeholder_texts.get(name, "")

        # Create label and entry box using the frame's method
        label, entry, variable = frame.create_label_and_entry_box(
            name,
            font,
            rel_width=size,
            placeholder_text=placeholder_text,
            placeholder_text_color=placeholder_color,
        )

        # Configure text color for the label and entry
        label.configure(text_color=TEXT_COLOR)
        entry.configure(text_color=TEXT_COLOR)

        # Store created label, entry, and variable
        labels.append(label)
        entries.append(entry)
        variables[name] = variable

    return labels, entries, variables


def create_section_widgets(
    title_frame: gui.Frame,
    data_frame: gui.Frame,
    title: str,
    entry_names: List[str],
    entry_sizes: List[float],
    placeholder_texts: Dict[str, str],
    checkbox_name: str,
    checkbox_func: Callable,
) -> Tuple[ctk.CTkLabel, List[ctk.CTkLabel], List[tk.Widget], Dict[str, Any]]:
    """
    Create and configure section widgets including labels, entries, a title label,
    and a checkbox, and return them along with a dictionary of input variables.

    Parameters
    ----------
    title_frame : gui.Frame
        The frame where the title label will be created.
    data_frame : gui.Frame
        The frame where the labels, entries, and checkbox will be created.
    title : str
        The text to display in the title label.
    entry_names : List[str]
        The names for the labels and entries.
    entry_sizes : List[float]
        The relative width sizes for the entries.
    placeholder_texts : Dict[str, str]
        A dictionary mapping names to placeholder texts for the entries.
    checkbox_name : str
        The name for the checkbox label.
    checkbox_func : Callable
        The function to be called when the checkbox state changes.

    Returns
    -------
    Tuple[gui.Label, List[gui.Label], List[gui.Widget], Dict[str, gui.Variable]]
        A tuple containing:
        - The created title label.
        - A list of created labels.
        - A list of created entries and checkbox.
        - A dictionary mapping names to the created entry and checkbox variables.
    """
    labels = []
    entries = []
    input_data = {}

    # Create labels, entries, and their variables
    labels, entries, variables = create_labels_and_entries(
        data_frame,
        entry_names,
        entry_sizes,
        BODY_FONT,
        placeholder_texts=placeholder_texts,
        placeholder_color=PLACEHOLDER_COLOR,
    )
    input_data = variables

    # Create and place the title label
    title_label = title_frame.create_label(title, SUBTITLE_FONT)
    title_label.configure(text_color=ACCENT_COLOR)

    # Configure text colors for labels
    for label in labels:
        label.configure(text_color=TEXT_COLOR)

    # Create label and checkbox
    label, checkbox, var = data_frame.create_label_and_check_box(
        checkbox_name, BODY_FONT, command=checkbox_func
    )
    input_data[checkbox_name] = var
    labels.append(label)
    entries.append(checkbox)

    return title_label, labels, entries, input_data


def place_section_widgets(
    title_frame,
    data_frame,
    title_label,
    labels,
    widgets,
    rel_padding,
    max_labels_per_row=1,
    hidden_fields=None,
):
    # Remove any widgets already there
    for widget in data_frame.winfo_children():
        widget.grid_forget()

    # Skip hidden fields
    if hidden_fields is not None:
        new_labels, new_widgets = [], []
        for label, widget in zip(labels, widgets):
            if label.cget("text") not in hidden_fields:
                new_labels.append(label)
                new_widgets.append(widget)
        labels, widgets = new_labels, new_widgets

    # Draw title
    title_frame.center_widget(title_label)
    # Align labelled widgets on grid
    data_frame.align_labelled_widgets_on_grid(
        labels,
        widgets,
        max_labels_per_row=max_labels_per_row,
        rel_padding=rel_padding,
    )


def generate_intervals_func():

    if independent_input_data[INDEPENDENT_CHECKBOX_NAME].get() == 1:
        place_independent_widgets(hidden_fields=None)
    else:
        for entry in independent_hidden_entries.values():
            entry.delete(0, ctk.END)
        place_independent_widgets(hidden_fields=INDEPENDENT_HIDDEN_FIELDS)


def multiple_readings_func():
    if dependent_input_data[DEPENDENT_CHECKBOX_NAME].get() == 1:
        place_dependent_widgets(hidden_fields=DEPENDENT_HIDDEN_FIELDS_1)
        dependent_hidden_entries["Uncertainty"].delete(0, ctk.END)
    else:
        place_dependent_widgets(hidden_fields=DEPENDENT_HIDDEN_FIELDS_2)
        dependent_hidden_entries["No. Repeats"].delete(0, ctk.END)


def output_error_message(message: str, section: str = "", color: str = None):
    color = ERROR_COLOR if color is None else color
    text_box.configure(text_color=color)
    if section == "":
        text_box.enter_text(message, center=True)
    else:
        text_box.enter_text(f"{section} - {message}", center=True)


def clear_error_box():
    text_box.enter_text("")


def separate_unit_and_prefix(full_unit: str):
    # Check for special case of g
    if full_unit == "g":
        return "kg", 1e-3 # Scale by 1e-3 to get back to kg
    
    if len(full_unit) == 1:
        return full_unit, 1
    
    # Check for any other g such as kg or mg
    if full_unit[1] == "g":
        return "kg", equations.prefix_lookup[full_unit[0]] * 1e-3      

    return full_unit[1], equations.prefix_lookup[full_unit[0]]


def get_user_experiment_details():
    # Get the data as a dict and perform all error checks
    independent_data = {name: var.get() for name, var in independent_input_data.items()}
    independent_data = validator.validate_dict(
        data=independent_data,
        dtypes=INDEPENDENT_DATA_TYPES,
        error_function=partial(output_error_message, section="Independent Data"),
        default_values=INDEPENDENT_DEFAULT_VALUES,
        additional_checks=INDEPENDENT_DATA_CHECKS,
    )

    if independent_data is None:
        return (None, None, None)

    dependent_data = {name: var.get() for name, var in dependent_input_data.items()}
    dependent_data = validator.validate_dict(
        data=dependent_data,
        dtypes=DEPENDENT_DATA_TYPES,
        error_function=partial(output_error_message, section="Dependent Data"),
        default_values=DEPENDENT_DEFAULT_VALUES,
        additional_checks=DEPENDENT_DATA_CHECKS,
    )

    if dependent_data is not None:
        multiple_readings = dependent_data[DEPENDENT_CHECKBOX_NAME]
        num_repeats = dependent_data["No. Repeats"]
        if multiple_readings and (num_repeats <= 1 or num_repeats >= 9):
            output_error_message(
                "No. Readings: Expected a value greater than one and less than or equal to 8",
                section="Dependent Data",
            )
            dependent_data = None

    if dependent_data is None:
        return (None, None, None)

    graph_data = {name: var.get() for name, var in graph_input_data.items()}
    graph_data = validator.validate_dict(
        data=graph_data,
        dtypes=GRAPH_DATA_TYPES,
        error_function=partial(output_error_message, section="Graph Data"),
        default_values=GRAPH_DEFAULT_VALUES,
        additional_checks=GRAPH_DATA_CHECKS,
    )

    # Comparison errors
    if graph_data is not None:

        num_readings, readings_to_remove = (
            independent_data["No. Readings"],
            validator.get_readings_to_delete(graph_data["Remove Readings"]),
        )
        graph_data["Remove Readings"] = validator.get_readings_to_delete(
            graph_data["Remove Readings"]
        )

        if graph_data["Solve For"] not in graph_data["Gradient Formula"]:
            output_error_message(
                "Solve For: You must choose a symbol in your specified formula",
                section="Graph Data",
            )
            graph_data = None

        elif (
            graph_data["Calculate Intercept"] and graph_data["Set Origin as Intercept"]
        ):
            output_error_message(
                (
                    "Calculate Intercept: No point calculating the intercept"
                    + " when the origin is set as the intercept"
                ),
                section="Graph Data",
            )
            graph_data = None

        elif any(
            reading > num_readings or reading < 1 for reading in readings_to_remove
        ):
            output_error_message(
                (
                    "Remove Readings: Reading numbers to remove cannot be greater than"
                    + f" {num_readings} or less than 1"
                ),
                section="Graph Data",
            )
            graph_data = None

        elif len(readings_to_remove) >= num_readings - 1:
            output_error_message(
                (
                    "Remove Readings: Less than two points would be left so a graph"
                    + " can't be made, reduce the number of readings removed"
                ),
                section="Graph Data",
            )
            graph_data = None

    if graph_data is None:
        return (None, None, None)

    # Split up the prefix and unit
    full_unit = independent_data["Unit"]
    independent_data["Single Unit"], independent_data["Prefix"] = (
        separate_unit_and_prefix(full_unit)
    )

    full_unit = dependent_data["Unit"]
    dependent_data["Single Unit"], dependent_data["Prefix"] = separate_unit_and_prefix(
        full_unit
    )

    clear_error_box()

    return independent_data, dependent_data, graph_data


def place_constants_section(equation: str, var_to_solve: str):
    global constants_data, is_constants
    constants = equations.get_all_constants(equation, var_to_solve)

    # Check if anything has changed
    if constants and is_constants and len(constants) == len(constants_data):
        return

    # If there is no constants section placed
    if constants_frame.winfo_children():
        for widget in constants_frame.winfo_children():
            widget.destroy()

    # Create and place constants title
    constants_title_label = constants_title_frame.create_label(
        CONSTANTS_TITLE, SUBTITLE_FONT
    )
    constants_title_label.configure(text_color=TEXT_COLOR)
    constants_title_frame.center_widget(constants_title_label)

    # If there are no constants
    if not constants:
        # Place no constants text
        constants_frame.center_widget(
            widget=constants_frame.create_label(
                text=NO_CONSTANTS_TEXT, font=SMALL_TITLE_FONT
            )
        )
        is_constants = False

    else:
        labels, entries, variables = create_labels_and_entries(
            constants_frame,
            names=constants,
            sizes=(CONSTANTS_ENTRY_SIZE,) * len(constants),
            font=SMALL_TITLE_FONT,
            placeholder_texts={constant: "e.g. 10 +/- 1" for constant in constants},
            placeholder_color=PLACEHOLDER_COLOR,
        )

        constants_frame.align_labelled_widgets_on_grid(
            labels=labels,
            widgets=entries,
            max_labels_per_row=len(labels),
            rel_padding=UI_PADDING,
        )

        constants_data = variables
        is_constants = True


def get_constants_values():
    constants = {symbol: var.get() for symbol, var in constants_data.items()}
    for symbol, value in constants.items():
        constants[symbol] = validator.get_ufloat(
            input_str=value,
            variable=symbol,
            error_func=partial(output_error_message, section="Constants"),
        )

        if constants[symbol] <= 0:
            output_error_message(
                message=(
                    f"{symbol}: Expected a value greater than zero, if you need"
                    + ' negative measurements, use the "-" sign in the formula'
                ),
                section="Constants"
            )
            plt.close()
            return None

        if constants[symbol] is None:
            plt.close()
            return None
    return constants


def generate_table():
    # Get user input and leave if there are any errors
    independent_data, dependent_data, graph_data = get_user_experiment_details()
    if independent_data is None:
        return
    equation, var_to_solve = graph_data["Gradient Formula"], graph_data["Solve For"]
    place_constants_section(equation, var_to_solve)

    global state
    if state == State.No_Table:
        # Delete old button
        generate_table_btn.configure(text=NEW_GENERATE_BTN_NAME)
        # Draw new button positions
        buttons_frame.align_widgets(
            widgets=[generate_table_btn, save_readings_btn],
            placements=EDITING_BUTTON_POSITIONS,
            orientation=gui.HORIZONTAL,
        )
        state = State.Editing

    elif state == State.Editing:
        # Delete table before making new one
        table = table_frame.winfo_children()[0]
        table.destroy()

    elif state == State.Finished:
        # Delete table before making new one
        table = table_frame.winfo_children()[0]
        table.destroy()
        re_enter_readings_btn.place_forget()
        generate_graph_btn.place_forget()
        # Draw new button positions
        buttons_frame.align_widgets(
            widgets=[generate_table_btn, save_readings_btn],
            placements=EDITING_BUTTON_POSITIONS,
            orientation=gui.HORIZONTAL,
        )
        state = State.Editing

    table_frame.fill_frame(
        widget=table_frame.create_table(
            generator.convert_to_array(
                generator.create_dataframe(
                    generator.get_initial_table_from_details(
                        independent_data, dependent_data
                    )
                )
            ),
            header_color=BG_COLOR,
            editable=True,
        ),
        rel_padx=TABLE_UI_PADDING,
        rel_pady=TABLE_UI_PADDING,
    )


def re_enter_readings():
    global state
    # Destroy old table
    table = table_frame.winfo_children()[0]
    table.destroy()
    # Add the old table back
    table_frame.fill_frame(
        widget=table_frame.create_table(
            data=original_table,
            header_color=BG_COLOR,
            editable=True,
        ),
        rel_padx=TABLE_UI_PADDING,
        rel_pady=TABLE_UI_PADDING,
    )
    # Revert buttons back to generate and save
    re_enter_readings_btn.place_forget()
    generate_graph_btn.place_forget()
    # Draw new button positions
    buttons_frame.align_widgets(
        widgets=[generate_table_btn, save_readings_btn],
        placements=EDITING_BUTTON_POSITIONS,
        orientation=gui.HORIZONTAL,
    )
    state = State.Editing
    clear_error_box()


def save_readings():
    # Get table to store old data
    global original_table, state
    independent_data, dependent_data, graph_data = get_user_experiment_details()
    # Don't proceed if errors are spotted
    if independent_data is None:
        return
    # Get table data and number of repeats
    num_repeats = dependent_data["No. Repeats"]
    table = table_frame.winfo_children()[0]
    # Save the data from the original table
    original_table = table.get()
    # Get data in table
    try:
        data_frame = generator.create_dataframe(
            data=(
                tuple(map(float, row)) if row_num != 0 else row
                for row_num, row in enumerate(original_table)
            ),
            headers_row=True,
        )
    except ValueError:
        output_error_message(
            "Table: Invalid entry, check all entered values are numbers"
        )
        return
    # Delete table from GUI
    table.destroy()
    # Replace constants section if needed
    equation, var_to_solve = graph_data["Gradient Formula"], graph_data["Solve For"]
    place_constants_section(equation, var_to_solve)
    # Original table always has two headers, the first one is independent data
    independent_header = original_table[0][0]

    independent_values = processor.get_data_frame_column(data_frame, independent_header)
    # Convert the floats to ufloats with uncertainties
    independent_ufloats = map(
        lambda x: ufloat(x, independent_data["Uncertainty"]), independent_values
    )
    # Multiply the prefix and apply the exponent
    independent_values_corrected = tuple(
        map(
            lambda x: (x * independent_data["Prefix"]) ** independent_data["Exponent"],
            independent_ufloats,
        )
    )
    # Get the uncertainty from the new calculated ufloat
    independent_uncertainties = tuple(x.std_dev for x in independent_values_corrected)

    # Calculate means of data if needed
    if num_repeats > 1:
        dependent_data_means = tuple(processor.calculate_means(data_frame))
        dependent_errs = tuple(processor.calculate_absolute_uncertainties(data_frame))
        dependent_ufloats = tuple(
            ufloat(x, err) for x, err in zip(dependent_data_means, dependent_errs)
        )
    else:
        # Original table always has two headers, the second one is dependent data
        dependent_header = original_table[0][1]
        dependent_data_means = processor.get_data_frame_column(
            data_frame, dependent_header
        )
        # Convert the floats to ufloats with uncertainties
        dependent_ufloats = map(
            lambda x: ufloat(x, dependent_data["Uncertainty"]), dependent_data_means
        )
    # Multiply the prefix and apply the exponent
    dependent_values_corrected = tuple(
        map(
            lambda x: (x * dependent_data["Prefix"]) ** dependent_data["Exponent"],
            dependent_ufloats,
        )
    )
    # Get the uncertainty from the new calculated ufloat
    dependent_uncertainties = tuple(x.std_dev for x in dependent_values_corrected)

    # Label it as (m^2) for example if an exponent of 2 is used
    simplifier = equations.simplify_float
    independent_data_unit = (
        f" ({independent_data['Single Unit']}^{simplifier(independent_data['Exponent'])})"
        if independent_data["Exponent"] != 1
        else f" ({independent_data['Single Unit']})"
    )

    dependent_data_unit = (
        f" ({dependent_data['Single Unit']}^{simplifier(dependent_data['Exponent'])})"
        if dependent_data["Exponent"] != 1
        else f" ({dependent_data['Single Unit']})"
    )

    # Start with the table headers
    new_table = [
        (
            independent_data["Name"] + independent_data_unit,
            independent_data["Name"] + " Uncertainty (+/-)",
            dependent_data["Name"] + dependent_data_unit,
            dependent_data["Name"] + " Uncertainty (+/-)",
        ),
    ]

    new_table.extend(
        zip(
            (
                equations.format_to_sig_figs(x.nominal_value)
                for x in independent_values_corrected
            ),
            map(equations.format_to_sig_figs, independent_uncertainties),
            (
                equations.format_to_sig_figs(x.nominal_value)
                for x in dependent_values_corrected
            ),
            map(equations.format_to_sig_figs, dependent_uncertainties),
        )
    )

    # Redraw Table
    table_frame.fill_frame(
        widget=table_frame.create_table(
            data=new_table,
            header_color=BG_COLOR,
            editable=False,
        ),
        rel_padx=TABLE_UI_PADDING,
        rel_pady=TABLE_UI_PADDING,
    )

    # Remove old buttons
    save_readings_btn.place_forget()
    # Draw new button positions
    buttons_frame.align_widgets(
        widgets=[generate_table_btn, re_enter_readings_btn, generate_graph_btn],
        placements=SAVED_BUTTON_POSITIONS,
        orientation=gui.HORIZONTAL,
    )

    state = State.Finished


def generate_graph():
    global is_constants
    independent_data, dependent_data, graph_data = get_user_experiment_details()

    if graph_data is None:
        return

    equation, var_to_solve = graph_data["Gradient Formula"], graph_data["Solve For"]

    # If they added constants add space to enter them before generating the graph
    if not is_constants and equations.get_all_constants(equation, var_to_solve):
        place_constants_section(equation, var_to_solve)
        return

    # Create a dataframe from the data in the table
    table_values = table_frame.winfo_children()[0].get()
    data_frame = generator.create_dataframe(
        data=(
            tuple(map(float, row)) if row_num != 0 else row
            for row_num, row in enumerate(table_values)
        ),
        headers_row=True,
    )

    # Independent headers are on the first two columns
    independent_header, uncertainty_header = table_values[0][0], table_values[0][1]
    independent_errors = processor.get_data_frame_column(data_frame, uncertainty_header)

    # Dependent headers are on the last two columns
    dependent_header, uncertainty_header = table_values[0][2], table_values[0][3]
    dependent_errors = processor.get_data_frame_column(data_frame, uncertainty_header)

    # Winfo children gets all the widgets in a frame or window
    figure, plot = plotter.create_plot(*PLOT_SIZE)
    # Plot graph
    # Check x-axis and y-axis position
    if graph_data["Variable On X-Axis"] == "Independent":
        errors_x, errors_y = independent_errors, dependent_errors
        x_header, y_header = independent_header, dependent_header
    else:
        errors_x, errors_y = dependent_errors, independent_errors
        x_header, y_header = dependent_header, independent_header

    # Get data just for graphing
    x_values = processor.get_data_frame_column(data_frame, header=x_header)
    y_values = processor.get_data_frame_column(data_frame, header=y_header)

    # Delete any anomalies
    readings_to_remove = graph_data["Remove Readings"]
    if readings_to_remove:
        x_values = tuple(
            x for i, x in enumerate(x_values) if i + 1 not in readings_to_remove
        )
        y_values = tuple(
            y for i, y in enumerate(y_values) if i + 1 not in readings_to_remove
        )
        errors_x = tuple(
            x for i, x in enumerate(errors_x) if i + 1 not in readings_to_remove
        )
        errors_y = tuple(
            y for i, y in enumerate(errors_y) if i + 1 not in readings_to_remove
        )

    hide_origin = not graph_data["Show Origin"]
    is_origin_intercept = graph_data["Set Origin as Intercept"]

    x, y = x_values, y_values
    # Gradient is negative and the origin is the intercept
    if ((y[0] - y[1]) * (x[0] - x[1])) < 0 and is_origin_intercept:
        output_error_message(
            message=(
                "Set Origin as Intercept: If origin is the intercept then the gradient"
                + " must be positive not negative"
            ),
            section="Graph Data",
        )
        plt.close()
        return

    # Plot scatter graph with a line of best fit on graph
    mean_gradient, mean_intercept = plotter.plot_scatter_graph(
        plot,
        x=x_values,
        y=y_values,
        title=(
            f"Investigating {independent_data['Name']} Against {dependent_data['Name']}",
            MEDIUM_FONT,
        ),
        axis_labels=(x_header, y_header),
        size=MARKER_SIZE,
        marker_color=MARKER_COLOR,
        line_color=LINE_COLOR,
        line_width=LINE_WIDTH,
        is_origin_intercept=is_origin_intercept,
        hide_origin=hide_origin,
    )  # TODO Change title

    if graph_data["Show Uncertainties"]:
        # Then add the error bars
        plotter.add_error_bars(
            plot,
            x=x_values,
            y=y_values,
            errors_x=errors_x,
            errors_y=errors_y,
            color=MARKER_COLOR,
            capsize=ERROR_BAR_CAP_SIZE,
            alpha=ERROR_BAR_ALPHA,
        )

        # Calculate the max and min gradients
        x_max, y_max = processor.calculate_max_gradient_points(
            x=x_values,
            y=y_values,
            errors_y=np.array(errors_y),
            errors_x=np.array(errors_x),
            is_origin_intercept=is_origin_intercept,
        )
        x_min, y_min = processor.calculate_min_gradient_points(
            x=x_values,
            y=y_values,
            errors_y=np.array(errors_y),
            errors_x=np.array(errors_x),
            is_origin_intercept=is_origin_intercept,
        )

        # Plot the min and max gradient points
        max_gradient, max_intercept = plotter.add_best_fit_line(
            plot,
            x_max,
            y_max,
            color=LINE_COLOR,
            line_style="--",
            alpha=UNCERTAINTY_LINE_ALPHA,
            line_width=LINE_WIDTH,
            is_origin_intercept=is_origin_intercept,
            hide_origin=hide_origin,
        )
        min_gradient, min_intercept = plotter.add_best_fit_line(
            plot,
            x_min,
            y_min,
            color=LINE_COLOR,
            line_style="--",
            alpha=UNCERTAINTY_LINE_ALPHA,
            line_width=LINE_WIDTH,
            is_origin_intercept=is_origin_intercept,
            hide_origin=hide_origin,
        )

        gradient_value = ufloat(mean_gradient, abs(max_gradient - min_gradient) / 2)

        text = (
            "Gradient: "
            + f"{equations.format_to_sig_figs(gradient_value.nominal_value)} +/- "
            + f"{equations.format_to_sig_figs(gradient_value.std_dev)}\n"
        )

        if graph_data["Calculate Intercept"]:
            text += (
                "Intercept: "
                + f"{equations.format_to_sig_figs(mean_intercept)} +/- "
                + f"{equations.format_to_sig_figs(abs(max_intercept - min_intercept) / 2)}\n"
            )

    else:
        gradient_value = ufloat(mean_gradient, 0)
        text = (
            "Gradient = "
            + f"{equations.format_to_sig_figs(gradient_value.nominal_value)}\n"
        )

        if graph_data["Calculate Intercept"]:
            text = "Intercept = " + f"{equations.format_to_sig_figs(mean_intercept)}\n"

    # If there are constants
    if equations.get_all_constants(equation, var_to_solve):
        constants = get_constants_values()
        # If an error happened
        if constants is None:
            return
    else:
        constants = None

    solutions = calculate_variable_using_gradient(
        graph_data["Gradient Formula"],
        graph_data["Solve For"],
        gradient_value,
        constants,
        show_uncertainties=graph_data["Show Uncertainties"],
    )

    for sol in solutions:
        text += sol + "\n"

    output_error_message(
        message=equations.get_expression(
            graph_data["Gradient Formula"], graph_data["Solve For"]
        ),
        section="Formula Used",
        color=TEXT_COLOR,
    )

    plotter.plot_text(
        plot, text=text, pos=GRADIENT_TEXT_POS, font_size=XSMALL_FONT, color=TEXT_COLOR
    )

    plt.show()


def calculate_variable_using_gradient(
    gradient_eq: str,
    var_to_solve: str,
    gradient_value: ufloat,
    constants: Dict[str, ufloat] = None,
    show_uncertainties: bool = True,
) -> Generator:
    solutions = equations.solve_equation(
        gradient_eq, var_to_solve, gradient_value, constants
    )

    if not solutions:
        yield f"There are no real solutions for {var_to_solve}"

    for sol in solutions:
        uncertainty_text = (
            f" +/- {equations.format_to_sig_figs(sol.std_dev)}"
            if show_uncertainties
            else ""
        )
        yield (
            f"{var_to_solve} = {equations.format_to_sig_figs(sol.nominal_value)}"
            + uncertainty_text
        )


if __name__ == "__main__":
    ctypes.windll.shcore.SetProcessDpiAwareness(1)

    # CONSTANTS
    SIZE = (1000, 750)
    TITLE = "CPAC Logger"

    # Fonts
    TYPE_SCALE = 4 / 3
    BASE_FONT = 18

    XSMALL_FONT = BASE_FONT / (TYPE_SCALE**2)
    SMALL_FONT = BASE_FONT / TYPE_SCALE
    MEDIUM_FONT = BASE_FONT
    LARGE_FONT = BASE_FONT * TYPE_SCALE
    XLARGE_FONT = BASE_FONT * TYPE_SCALE**2

    TITLE_FONT = ("Segoe UI Bold", XLARGE_FONT)
    SUBTITLE_FONT = ("Segoe UI Bold", MEDIUM_FONT)
    SMALL_TITLE_FONT = ("Segoe UI", SMALL_FONT)
    BODY_FONT = ("Segoe UI", XSMALL_FONT)

    # Colors
    BG_COLOR = "#242933"
    PRIMARY_COLOR = "#80bef4"
    SECONDARY_COLOR = "#393E46"
    TEXT_COLOR = "#ebebeb"
    ACCENT_COLOR = "#37E9E3"
    PLACEHOLDER_COLOR = "#A3A3A3"

    # Proportions
    UI_TO_TABLE_SPLIT = (0.3, 0.7)
    UI_PADDING = 0.008
    # (Title, SubTitle, Independent Data, SubTitle, Dependent Data, Subtitle, Graph Data)
    UI_SPLIT = (0.07, 0.06, 0.27, 0.06, 0.20, 0.06, 0.28)
    # Table Header, Buttons, Table, Constants Title, Constants, Error Box and Title
    TABLE_OUTPUT_SPLIT = (0.07, 0.05, 0.61, 0.06, 0.03, 0.18)

    # Independent Section
    INDEPENDENT_TITLE = "Independent Variable"
    INDEPENDENT_ENTRY_NAMES = [
        "Name",
        "Unit",
        "No. Readings",
        "Uncertainty",
        "Exponent",
        "Interval Size",
        "Starting Value",
    ]
    INDEPENDENT_PLACEHOLDER_TEXTS = {
        "Name": "e.g. Force",
        "Unit": "e.g. kN",
        "No. Readings": "e.g. 5",
        "Uncertainty": "0",
        "Exponent": "1",
    }
    INDEPENDENT_ENTRY_SIZES = [0.1, 0.045, 0.04, 0.05, 0.05, 0.05, 0.05]
    INDEPENDENT_DATA_TYPES = [str, str, int, float, validator.exponent, float, float]
    INDEPENDENT_DEFAULT_VALUES = {
        "Uncertainty": 0,
        "Interval Size": 0,
        "Starting Value": 0,
        "Exponent": "1",
    }
    INDEPENDENT_HIDDEN_FIELDS = ("Interval Size", "Starting Value")

    INDEPENDENT_DATA_CHECKS = {
        "Name": partial(validator.is_letters_and_spaces, field="Name"),
        "Unit": validator.validate_unit,
        "No. Readings": lambda x: (
            f"No. Readings: {x} must be between 1 and 15 inclusive"
            if x <= 0 or x > 15
            else None
        ),
        "Uncertainty": lambda x: (
            f'Uncertainty: "{x}" is not valid, it must be greater or equal to 0'
            if x < 0
            else None
        ),
        "Exponent": lambda x: (
            f'Exponent: "{x}" must be a non-zero fraction or decimal'
            if x == 0
            else None
        ),
    }

    INDEPENDENT_CHECKBOX_NAME = "Generate Intervals"
    INDEPENDENT_CHECKBOX_FUNCTION = generate_intervals_func

    # Dependant Section
    DEPENDENT_TITLE = "Dependent Variable"
    DEPENDENT_ENTRY_NAMES = ["Name", "Unit", "No. Repeats", "Uncertainty", "Exponent"]
    DEPENDENT_PLACEHOLDER_TEXTS = {
        "Name": "e.g. Extension",
        "Unit": "e.g. cm",
        "No. Repeats": "e.g. 3",
        "Uncertainty": "0",
        "Exponent": "1",
    }
    DEPENDENT_ENTRY_SIZES = [0.1, 0.045, 0.04, 0.05, 0.05]
    DEPENDENT_DATA_TYPES = [str, str, int, float, validator.exponent]
    DEPENDENT_DEFAULT_VALUES = {
        "No. Repeats": 1,
        "Uncertainty": 0,
        "Exponent": "1",
    }
    DEPENDENT_HIDDEN_FIELDS_1 = ("Uncertainty",)
    DEPENDENT_HIDDEN_FIELDS_2 = ("No. Repeats",)
    DEPENDENT_HIDDEN_FIELDS = ("Uncertainty", "No. Repeats")

    DEPENDENT_DATA_CHECKS = {
        "Name": partial(validator.is_letters_and_spaces, field="Name"),
        "Unit": validator.validate_unit,
        "No. Repeats": lambda x: (
            f"No. Repeats: {x} must must be between 1 and 10 inclusive"
            if x < 1 and x > 10
            else None
        ),
        "Uncertainty": lambda x: (
            f'Uncertainty: "{x}" is not valid, it must be greater or equal to 0'
            if x < 0
            else None
        ),
    }

    DEPENDENT_CHECKBOX_NAME = "Multiple Readings"
    DEPENDENT_CHECKBOX_FUNCTION = multiple_readings_func

    # Graph
    GRAPH_TITLE = "Graph Options"
    GRAPH_ENTRY_NAMES = ["Gradient Formula", "Solve For", "Remove Readings"]
    GRAPH_PLACEHOLDER_TEXTS = {
        "Gradient Formula": "e.g. (T/u)^(1/2)",
        "Solve For": "e.g. u",
        "Remove Readings": "e.g. 1, 2, 5",
    }
    GRAPH_ENTRY_SIZES = [0.1, 0.04, 0.06]
    GRAPH_OPTION_MENU_NAME = "Variable On X-Axis"
    GRAPH_OPTION_MENU_VALUES = ("Independent", "Dependent")
    GRAPH_CHECKBOX_NAMES = (
        "Show Uncertainties",
        "Show Origin",
        "Set Origin as Intercept",
        "Calculate Intercept",
    )
    GRAPH_CHECKED_BOXES = ("Show Uncertainties", "Show Origin")
    GRAPH_DATA_TYPES = [str, str, str, str, bool, bool, bool]
    GRAPH_DEFAULT_VALUES = {"Remove Readings": ""}
    GRAPH_DATA_CHECKS = {
        "Gradient Formula": validator.validate_formula,
        "Solve For": partial(validator.is_letters_and_spaces, field="Solve For"),
        "Remove Readings": validator.is_readings_input_correct,
    }

    # Constants
    CONSTANTS_TITLE = "Constants"
    CONSTANTS_ENTRY_SIZE = 0.09
    NO_CONSTANTS_TEXT = "No constants need to be entered"

    # BUTTONS
    BUTTON_NAMES = [
        "Generate Table",
        "Re-Enter Readings",
        "Save Readings",
        "Generate Graph",
    ]
    BUTTON_FUNCTIONS = [
        generate_table,
        re_enter_readings,
        save_readings,
        generate_graph,
    ]
    NEW_GENERATE_BTN_NAME = "Generate New Table"
    NO_TABLE_BUTTON_POSITION = 0.5
    EDITING_BUTTON_POSITIONS = [0.42, 0.58]
    SAVED_BUTTON_POSITIONS = [0.34, 0.5, 0.66]
    BUTTON_SIZE = 0.08

    # Table
    TABLE_TITLE = "Experiment Table"
    TABLE_UI_PADDING = 0.015

    # Text box
    TEXT_BOX_TITLE = "Errors"
    TEXT_BOX_SIZE = [0.6, 0.08]
    TEXT_BOX_PLACEMENTS = [0.15, 0.55]
    ERROR_COLOR = "#FC6F40"

    # Plot
    PLOT_SIZE = (9, 7)
    PLOT_SCALE = 3.125
    UNCERTAINTY_LINE_ALPHA = 0.3
    MARKER_SIZE = 25
    LINE_COLOR = "#7BA2FF"
    MARKER_COLOR = "#8FFFFB"
    ERROR_BAR_ALPHA = 0.5
    LINE_WIDTH = 1.5
    ERROR_BAR_CAP_SIZE = 2
    GRADIENT_TEXT_POS = (0.5, 0.8)

    # Global variables
    state = State.No_Table
    constants_data = {}
    is_constants = False

    # Global table variable to store old table or re editing
    original_table = ()

    # Create window
    gui.set_app_size(*SIZE)
    app = gui.App(TITLE)
    app.resizable(width=False, height=False)
    # Setup figure and plot and scale window so they are similar in size
    plt.style.use(
        "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle"
    )
    matplotlib.use( 'tkagg' )
    app.tk.call("tk", "scaling", PLOT_SCALE)

    # Split window into two frames to hold the UI and table
    left_frame, right_frame = app.split_into_sub_frames(
        UI_TO_TABLE_SPLIT, gui.HORIZONTAL
    )
    right_frame.configure(fg_color=SECONDARY_COLOR)
    left_frame.configure(fg_color=BG_COLOR)

    # Split left frame into all the UI frames
    (
        title_frame,
        independent_title_frame,
        independent_data_frame,
        dependent_title_frame,
        dependent_data_frame,
        graph_title_frame,
        graph_data_frame,
    ) = left_frame.split_into_sub_frames(UI_SPLIT, gui.VERTICAL)

    # Create and place title
    title_label = title_frame.create_label(TITLE, TITLE_FONT)
    title_label.configure(text_color=TEXT_COLOR)
    title_frame.center_widget(title_label)

    # Create all widgets needed
    title_label, labels, widgets, independent_input_data = create_section_widgets(
        title_frame=independent_title_frame,
        data_frame=independent_data_frame,
        title=INDEPENDENT_TITLE,
        entry_names=INDEPENDENT_ENTRY_NAMES,
        entry_sizes=INDEPENDENT_ENTRY_SIZES,
        placeholder_texts=INDEPENDENT_PLACEHOLDER_TEXTS,
        checkbox_name=INDEPENDENT_CHECKBOX_NAME,
        checkbox_func=INDEPENDENT_CHECKBOX_FUNCTION,
    )

    # Create partial functions for widget placement
    place_independent_widgets = partial(
        place_section_widgets,
        title_frame,
        independent_data_frame,
        title_label,
        labels,
        widgets,
        UI_PADDING,
    )

    place_independent_widgets(hidden_fields=INDEPENDENT_HIDDEN_FIELDS)

    independent_hidden_entries = {
        label.cget("text"): entry
        for label, entry in zip(labels, widgets)
        if label.cget("text") in INDEPENDENT_HIDDEN_FIELDS
    }

    # Create all widgets needed
    title_label, labels, widgets, dependent_input_data = create_section_widgets(
        title_frame=dependent_title_frame,
        data_frame=dependent_data_frame,
        title=DEPENDENT_TITLE,
        entry_names=DEPENDENT_ENTRY_NAMES,
        entry_sizes=DEPENDENT_ENTRY_SIZES,
        placeholder_texts=DEPENDENT_PLACEHOLDER_TEXTS,
        checkbox_name=DEPENDENT_CHECKBOX_NAME,
        checkbox_func=DEPENDENT_CHECKBOX_FUNCTION,
    )

    # Create partial functions for widget placement
    place_dependent_widgets = partial(
        place_section_widgets,
        title_frame,
        dependent_data_frame,
        title_label,
        labels,
        widgets,
        UI_PADDING,
    )

    place_dependent_widgets(hidden_fields=DEPENDENT_HIDDEN_FIELDS_2)

    # Get the no. repeats entry to change placeholder text
    dependent_hidden_entries = {
        label.cget("text"): entry
        for label, entry in zip(labels, widgets)
        if label.cget("text") in DEPENDENT_HIDDEN_FIELDS
    }

    # Create and place graph section
    graph_title_label = graph_title_frame.create_label(GRAPH_TITLE, SUBTITLE_FONT)
    graph_title_label.configure(text_color=ACCENT_COLOR)
    graph_title_frame.center_widget(graph_title_label)

    graph_entry_names = GRAPH_ENTRY_NAMES
    graph_entry_sizes = GRAPH_ENTRY_SIZES

    graph_labels, graph_widgets, graph_input_data = create_labels_and_entries(
        graph_data_frame,
        graph_entry_names,
        graph_entry_sizes,
        font=BODY_FONT,
        placeholder_texts=GRAPH_PLACEHOLDER_TEXTS,
        placeholder_color=PLACEHOLDER_COLOR,
    )

    # Variable option_menu
    x_axis_var_label, x_axis_option_menu, x_axis_var = (
        graph_data_frame.create_label_and_option_menu(
            GRAPH_OPTION_MENU_NAME, BODY_FONT, GRAPH_OPTION_MENU_VALUES, rel_width=0.1
        )
    )
    # Set default selection
    graph_input_data[GRAPH_OPTION_MENU_NAME] = x_axis_var
    x_axis_option_menu.set(GRAPH_OPTION_MENU_VALUES[0])
    graph_widgets.append(x_axis_option_menu)
    graph_labels.append(x_axis_var_label)

    # Check Boxes
    for name in GRAPH_CHECKBOX_NAMES:
        label, checkbox, var = graph_data_frame.create_label_and_check_box(
            name, BODY_FONT
        )
        graph_input_data[name] = var
        if name in GRAPH_CHECKED_BOXES:
            checkbox.select()
        label.configure(text_color=TEXT_COLOR)
        graph_labels.append(label)
        graph_widgets.append(checkbox)

    graph_data_frame.align_labelled_widgets_on_grid(
        graph_labels,
        graph_widgets,
        max_labels_per_row=1,
        rel_padding=UI_PADDING,
    )

    # Split right frame into table and output
    (
        table_title_frame,
        buttons_frame,
        table_frame,
        constants_title_frame,
        constants_frame,
        output_frame,
    ) = right_frame.split_into_sub_frames(TABLE_OUTPUT_SPLIT, gui.VERTICAL)

    # Create table buttons
    generate_table_btn, re_enter_readings_btn, save_readings_btn, generate_graph_btn = (
        tuple(
            buttons_frame.create_button(
                text=btn_name,
                font=BODY_FONT,
                on_click=btn_func,
                rel_width=BUTTON_SIZE,
                bg_color=PRIMARY_COLOR,
                text_color=BG_COLOR,
                hover_color=ACCENT_COLOR,
            )
            for btn_name, btn_func in zip(BUTTON_NAMES, BUTTON_FUNCTIONS)
        )
    )

    # Place starting button as generate table
    buttons_frame.center_widget(generate_table_btn)

    # Draw table title
    table_title_frame.center_widget(
        table_title_frame.create_label(text=TABLE_TITLE, font=SUBTITLE_FONT)
    )

    # Draw output section
    label, text_box = output_frame.create_label_and_text_box(
        text=TEXT_BOX_TITLE,
        font=SUBTITLE_FONT,
        rel_size=TEXT_BOX_SIZE,
        read_only=True,
    )
    text_box.configure(fg_color=BG_COLOR, text_color=ERROR_COLOR, font=SMALL_TITLE_FONT)
    output_frame.align_widgets(
        widgets=(label, text_box),
        placements=TEXT_BOX_PLACEMENTS,
        orientation=gui.VERTICAL,
    )

    app.mainloop()
