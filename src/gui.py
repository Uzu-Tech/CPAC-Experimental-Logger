import tkinter as tk
from typing import Any, Callable, Iterable, List, Tuple

import CTkTable as ctk_table
import customtkinter as ctk

# Constants
HORIZONTAL = "horizontal"
VERTICAL = "vertical"
APP_WIDTH = 500
APP_HEIGHT = 500


def set_app_size(width, height):
    global APP_WIDTH, APP_HEIGHT
    APP_WIDTH = width
    APP_HEIGHT = height


class GridPlacer:
    """
    Class for placing widgets on a grid.

    Methods
    -------
    place_widget_on_grid(widget, row=0, col=0, rel_padx=0, rel_pady=0, sticky=""):
        Places a widget on the grid.
    """

    def place_widget_on_grid(
        self,
        widget: tk.Widget,
        row: int = 0,
        col: int = 0,
        rel_padx: float = 0,
        rel_pady: float = 0,
        sticky: str = "",
    ) -> None:
        """
        Places a widget on the grid.

        Parameters
        ----------
        widget : tk.Widget
            The widget to place.
        row : int, optional
            The row position on the grid (default is 0).
        col : int, optional
            The column position on the grid (default is 0).
        rel_padx : float, optional
            The relative padding in the x direction (default is 0).
        rel_pady : float, optional
            The relative padding in the y direction (default is 0).
        sticky : str, optional
            Specifies which sides of the cell the widget sticks to (default is "").
        """
        widget.grid(
            row=row,
            column=col,
            padx=rel_padx * APP_WIDTH,
            pady=rel_pady * APP_HEIGHT,
            sticky=sticky,
        )


class Splittable:
    """
    Class for splitting frames into sub-frames.

    Methods
    -------
    split_into_sub_frames(proportions, orientation):
        Splits the frame into sub-frames based on given proportions and orientation.
    """

    def split_into_sub_frames(
        self, proportions: Tuple[float], orientation: str
    ) -> List["Frame"]:
        """
        Splits the frame into sub-frames based on given proportions and orientation.

        Parameters
        ----------
        proportions : Tuple[float]
            The proportions to split the frame.
        orientation : str
            The orientation for splitting ("horizontal" or "vertical").

        Returns
        -------
        List[Frame]
            A list of sub-frames.

        Raises
        ------
        ValueError
            If proportions do not add up to 1 or orientation is invalid.
        """
        if sum(proportions) != 1:
            raise ValueError("Proportions must add up to 1.")

        sub_frames = []
        total_weights = 100  # Scales up the weights
        if orientation == HORIZONTAL:
            for i, prop in enumerate(proportions):
                sub_frame = Frame(self, prop, 1)
                sub_frame.grid(row=0, column=i, sticky="")
                self.grid_rowconfigure(i, weight=int(prop * total_weights))
                sub_frames.append(sub_frame)
        elif orientation == VERTICAL:
            for i, prop in enumerate(proportions):
                sub_frame = Frame(self, 1, prop)
                sub_frame.grid(row=i, column=0, sticky="")
                self.grid_columnconfigure(i, weight=int(prop * total_weights))
                sub_frames.append(sub_frame)
        else:
            raise ValueError("Orientation must be either 'horizontal' or 'vertical'.")

        return sub_frames

    def resize_frames(
        self, frames: Tuple["Frame"], proportions: Tuple[float], orientation: str
    ):
        total_weights = 100  # Scales up the weights
        if orientation == HORIZONTAL:
            for i, prop in enumerate(proportions):
                sub_frame = frames[i]
                sub_frame.grid(row=0, column=i, sticky="")
                self.grid_rowconfigure(i, weight=int(prop * total_weights))
        elif orientation == VERTICAL:
            for i, prop in enumerate(proportions):
                sub_frame = frames[i]
                sub_frame.grid(row=i, column=0, sticky="")
                self.grid_columnconfigure(i, weight=int(prop * total_weights))
        else:
            raise ValueError("Orientation must be either 'horizontal' or 'vertical'.")


class App(ctk.CTk, GridPlacer, Splittable):
    """
    CustomTkinter Application class.

    Methods
    -------
    __init__(title, theme="dark", color="blue"):
        Initializes the application.
    place_widget_on_grid(widget, row=0, col=0, rel_padx=0, rel_pady=0, sticky=""):
        Places a widget on the grid.
    split_into_sub_frames(proportions, orientation):
        Splits the frame into sub-frames based on given proportions and orientation.
    """

    def __init__(self, title, theme="dark", color="blue"):
        """
        Initializes the application.

        Parameters
        ----------
        title : str
            The title of the application window.
        theme : str, optional
            The appearance mode ("dark", "light", or "system") (default is "dark").
        color : str, optional
            The color theme ("blue", "green", or "dark-blue") (default is "blue").
        """
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme(color)
        super().__init__()
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.width = APP_WIDTH
        self.height = APP_HEIGHT
        self.geometry(f"{str(APP_WIDTH)}x{str(APP_HEIGHT)}")
        self.title(title)


class Frame(ctk.CTkFrame, GridPlacer, Splittable):
    """
    Custom frame class for creating and managing widgets.

    Methods
    -------
    __init__(master, rel_width, rel_height, fg_color="transparent"):
        Initializes the frame.
    place_widget_on_grid(widget, row=0, col=0, rel_padx=0, rel_pady=0, sticky=""):
        Places a widget on the grid.
    split_into_sub_frames(proportions, orientation):
        Splits the frame into sub-frames based on given proportions and orientation.
    create_label(text, font):
        Creates a label widget.
    create_label_and_entry_box(text, font, variable, rel_width):
        Creates a label and entry box widget.
    create_label_and_check_box(text, font, variable):
        Creates a label and check box widget.
    create_label_and_combo_box(text, font, values, on_click=None):
        Creates a label and combo box widget.
    create_button(text, font, on_click):
        Creates a button widget.
    create_table(data):
        Creates a table widget.
    create_label_and_text_box(text, font, rel_size, read_only=False):
        Creates a label and text box widget.
    align_labelled_widgets_on_grid(labels, widgets, max_labels_per_row, rel_padding):
        Aligns labeled widgets on the grid.
    center_widget(widget, offsetx=0, offsety=0):
        Centers a widget in the frame.
    align_widgets(widgets, placements, orientation=VERTICAL):
        Aligns widgets in the frame.
    fill_frame(widget, rel_padx, rel_pady):
        Fills the frame with a widget.
    """

    def __init__(
        self,
        master: App | ctk.CTkFrame,
        rel_width: float,
        rel_height: float,
        fg_color: str = "transparent",
    ):
        """
        Initializes the frame.

        Parameters
        ----------
        master : App or ctk.CTkFrame
            The master widget.
        rel_width : float
            The relative width of the frame.
        rel_height : float
            The relative height of the frame.
        fg_color : str, optional
            The foreground color of the frame (default is "transparent").
        """
        self.width = master.width * rel_width
        self.height = master.height * rel_height

        super().__init__(
            master=master,
            width=self.width,
            height=self.height,
            fg_color=fg_color,
            corner_radius=0,
        )
        self.grid_propagate(False)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def create_label(self, text: str, font: Tuple[str, int]):
        """
        Creates a label widget.

        Parameters
        ----------
        text : str
            The text for the label.
        font : Tuple[str, int]
            The font for the label.

        Returns
        -------
        ctk.CTkLabel
            The created label widget.
        """
        return ctk.CTkLabel(
            master=self,
            text=text,
            font=font,
        )

    def create_label_and_entry_box(
        self, text: str, font: str, rel_width: float, **kwargs
    ) -> Tuple[ctk.CTkLabel, ctk.CTkEntry, ctk.StringVar]:
        """
        Creates a label and entry box widget.

        Parameters
        ----------
        text : str
            The text for the label.
        font : str
            The font for the entry box.
        variable : ctk.StringVar
            The variable to bind to the entry box.
        rel_width : float
            The relative width of the entry box.

        Returns
        -------
        Tuple[ctk.CTkLabel, ctk.CTkEntry]
            The created label and entry box widgets.
        """
        entry = ctk.CTkEntry(
            master=self,
            font=font,
            width=rel_width * APP_WIDTH,
            height=font[1] * 1.8,
            **kwargs
        )
        text = self.create_label(text, font)
        return text, entry, entry

    def create_label_and_check_box(
        self, text: str, font: str, command: Callable = None
    ) -> Tuple[ctk.CTkLabel, ctk.CTkCheckBox, ctk.IntVar]:
        """
        Creates a label and check box widget.

        Parameters
        ----------
        text : str
            The text for the label.
        font : str
            The font for the check box.
        variable : ctk.IntVar
            The variable to bind to the check box.

        Returns
        -------
        Tuple[ctk.CTkLabel, ctk.CTkCheckBox]
            The created label and check box widgets.
        """
        variable = ctk.IntVar()
        check_box = ctk.CTkCheckBox(
            master=self,
            text="",
            variable=variable,
            checkbox_height=font[1],
            checkbox_width=font[1],
            border_width=(font[1] / 8),
            corner_radius=(font[1] / 4),
            command=command,
        )
        text = self.create_label(text, font)
        return text, check_box, variable

    def create_label_and_option_menu(
        self,
        text: str,
        font: str,
        values: Tuple[Any],
        rel_width: float,
        on_click: Callable = None,
    ) -> Tuple[ctk.CTkLabel, ctk.CTkComboBox, ctk.StringVar]:
        """
        Creates a label and combo box widget.

        Parameters
        ----------
        text : str
            The text for the label.
        font : str
            The font for the combo box.
        values : Tuple[Any]
            The values for the combo box.
        on_click : Callable, optional
            The callback function for the combo box (default is None).

        Returns
        -------
        Tuple[ctk.CTkLabel, ctk.CTkComboBox]
            The created label and combo box widgets.
        """
        variable = ctk.StringVar()
        option_menu = ctk.CTkOptionMenu(
            master=self,
            font=font,
            values=values,
            command=on_click,
            variable=variable,
            height=font[1] * 1.8,
            width=rel_width * APP_WIDTH,
        )
        text = self.create_label(text, font)
        return text, option_menu, variable

    def create_button(
        self,
        text: str,
        font: str,
        on_click: Callable,
        rel_width: float,
        bg_color: str = None,
        text_color: str = None,
        hover_color: str = None,
    ) -> ctk.CTkButton:
        """
        Creates a button widget.

        Parameters
        ----------
        text : str
            The text for the button.
        font : str
            The font for the button.
        on_click : Callable
            The callback function for the button.

        Returns
        -------
        ctk.CTkButton
            The created button widget.
        """
        return ctk.CTkButton(
            master=self,
            text=text,
            font=font,
            command=on_click,
            width=rel_width * APP_WIDTH,
            fg_color=bg_color,
            text_color=text_color,
            hover_color=hover_color,
        )

    def create_table(
        self, data: Tuple[Tuple], header_color: str = None, editable: bool = True
    ) -> "Table":
        """
        Creates a table widget.

        Parameters
        ----------
        data : Tuple[Tuple]
            The data for the table.

        Returns
        -------
        Table
            The created table widget.
        """
        return Table(
            master=self, data=data, header_color=header_color, editable=editable
        )

    def create_label_and_text_box(
        self,
        text: str,
        font: str,
        rel_size: Tuple[float, float],
        read_only: bool = False,
    ) -> Tuple[ctk.CTkLabel, "ReadOnlyTextBox"]:
        """
        Creates a label and text box widget.

        Parameters
        ----------
        text : str
            The text for the label.
        font : str
            The font for the text box.
        rel_size : Tuple[float, float]
            The relative size of the text box.
        read_only : bool, optional
            If True, creates a read-only text box (default is False).

        Returns
        -------
        Tuple[ctk.CTkLabel, ctk.CTkTextbox]
            The created label and text box widgets.
        """
        TextBox = ReadOnlyTextBox if read_only else ctk.CTkTextbox
        text_box = TextBox(
            master=self,
            font=font,
            width=rel_size[0] * APP_WIDTH,
            height=rel_size[1] * APP_HEIGHT,
            wrap="word",
        )
        text = self.create_label(text, font)
        return text, text_box

    def align_labelled_widgets_on_grid(
        self,
        labels: Iterable[ctk.CTkLabel],
        widgets: Iterable[tk.Widget],
        max_labels_per_row: int,
        rel_padding: float,
    ) -> None:
        """
        Aligns labeled widgets on the grid.

        Parameters
        ----------
        labels : Iterable[ctk.CTkLabel]
            The labels to align.
        widgets : Iterable[tk.Widget]
            The widgets to align.
        max_labels_per_row : int
            The maximum number of labels per row.
        rel_padding : float
            The relative padding between widgets.
        """
        for index in range(len(labels)):
            row_num = index // max_labels_per_row
            label_col_num = (
                index * 2 % (max_labels_per_row * 2)
            )  # Draw Labels on every even column
            widget_col_num = label_col_num + 1  # Draw widgets on every odd column
            self.place_widget_on_grid(
                labels[index],
                row=row_num,
                col=label_col_num,
                sticky="E",
                rel_padx=rel_padding,
            )
            self.place_widget_on_grid(
                widgets[index],
                row=row_num,
                col=widget_col_num,
                sticky="W",
                rel_padx=rel_padding,
            )

        for row in range(len(labels) // max_labels_per_row):
            self.rowconfigure(row, weight=1)

        for col in range(max_labels_per_row * 2):
            self.columnconfigure(col, weight=1)

    def center_widget(self, widget: tk.Widget, offsetx=0, offsety=0) -> None:
        """
        Centers a widget in the frame.

        Parameters
        ----------
        widget : tk.Widget
            The widget to center.
        offsetx : int, optional
            The x-axis offset (default is 0).
        offsety : int, optional
            The y-axis offset (default is 0).
        """
        widget.place(anchor="c", relx=0.5 + offsetx, rely=0.5 + offsety)

    def align_widgets(
        self,
        widgets: Iterable[tk.Widget],
        placements: Iterable[float],
        orientation: str = VERTICAL,
    ) -> None:
        """
        Aligns widgets in the frame vertically or horizontally

        Parameters
        ----------
        widgets : Iterable[tk.Widget]
            The widgets to align.
        placements : Iterable[float]
            The relative placements of the widgets.
        orientation : str, optional
            The orientation for alignment ("vertical" or "horizontal") (default is VERTICAL).
        """
        offsets = tuple(placement - 0.5 for placement in placements)
        if orientation == VERTICAL:
            for widget, offset in zip(widgets, offsets):
                self.center_widget(widget, offsety=offset)
        else:
            for widget, offset in zip(widgets, offsets):
                self.center_widget(widget, offsetx=offset)

    def fill_frame(self, widget: tk.Widget, rel_padx: int, rel_pady: int):
        """
        Fills the frame with a widget.

        Parameters
        ----------
        widget : tk.Widget
            The widget to fill the frame with.
        rel_padx : int
            The relative padding in the x direction.
        rel_pady : int
            The relative padding in the y direction.
        """
        self.place_widget_on_grid(
            widget, sticky="NSWE", rel_padx=rel_padx, rel_pady=rel_pady
        )


class Table(ctk_table.CTkTable):
    """
    Custom table class for creating and managing table widgets.

    Methods
    -------
    __init__(master, data):
        Initializes the table.
    """

    def __init__(
        self,
        master: "Frame",
        data: Tuple[Tuple],
        header_color: str = None,
        editable: bool = True,
    ):
        """
        Initializes the table.

        Parameters
        ----------
        master : Frame
            The master frame.
        data : Tuple[Tuple]
            The data for the table.
        """
        super().__init__(
            master=master,
            row=len(data),
            column=len(data[0]),
            values=data,
            write=int(editable),
            corner_radius=5,
            header_color=header_color,
        )

    @property
    def items(self):
        """
        Retrieves the items in the table.

        Returns
        -------
        Any
            The items in the table.
        """
        return self.get()

    @property
    def headers(self):
        """
        Retrieves the headers of the table.

        Returns
        -------
        Any
            The headers of the table.
        """
        return self.get_row(0)


class ReadOnlyTextBox(ctk.CTkTextbox):
    """
    Custom read-only text box class.

    Methods
    -------
    __init__(master, font, width, height):
        Initializes the read-only text box.
    enter_text(text):
        Enters text into the text box.
    """

    def __init__(self, master: "Frame", font: str, width: int, height: int, wrap: str):
        """
        Initializes the read-only text box.

        Parameters
        ----------
        master : Frame
            The master frame.
        font : str
            The font for the text box.
        width : int
            The width of the text box.
        height : int
            The height of the text box.
        """
        super().__init__(
            master=master, font=font, width=width, height=height, wrap="word"
        )
        self.configure(state="disabled")

    def enter_text(self, text: str, center: bool = False):
        """
        Enters text into the text box.

        Parameters
        ----------
        text : str
            The text to enter.
        """
        self.configure(state="normal")
        self.delete("0.0", "end")  # delete all text
        self.insert("0.0", text)
        if center:
            self.tag_config("center", justify="center")
            self.tag_add("center", "0.0", "end")
        self.configure(state="disabled")
