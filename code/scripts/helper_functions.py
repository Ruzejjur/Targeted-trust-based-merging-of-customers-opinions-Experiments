
# Matplotlib: Plotting library
import matplotlib.pyplot as plt

# os: Interact with the operating system
import os

# Pandas: Data manipulation and analysis
import pandas as pd

import logging
import sys
from pathlib import Path



def plot_posterior_distribution(variable_names, data, y_label, title, save_path, x_tick_labelsize=17, figsize=(7, 5)):
    """
    Plots and saves a bar chart of a posterior distribution.

    Args:
        variable_names (list): List of strings for x-tick labels.
        data (list or np.array): Data for the bar chart.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.
        save_path (str): Full path to save the figure (including filename and .eps extension).
        x_tick_labelsize (int, optional): Font size for x-axis tick labels. Defaults to 17.
        figsize (tuple, optional): Figure size (width, height). Defaults to (7, 5).
    """
    # Create a figure window with a single subplot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plotting the updated posterior distribution as a bar chart
    ax.bar(variable_names, data)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=15.5)
    ax.tick_params(axis='x', labelsize=x_tick_labelsize)

    # Ensure the directory for saving exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the figure in .eps format
    plt.tight_layout()
    plt.savefig(save_path, format='eps')

    # Show the plot
    plt.show()

def format_and_display_percentages(data, column_names, header_text, decimal_places=2):
    """
    Formats numerical array values into percentage strings,
    creates a pandas DataFrame, and prints a header before displaying it.

    Args:
        data (list or np.array): List or array of numerical values.
        column_names (list): List of strings for DataFrame column headers.
        header_text (str): Text to print before displaying the DataFrame.
        decimal_places (int, optional): Number of decimal places for the percentage. Defaults to 2.


    Returns:
        pd.DataFrame: The DataFrame containing the formatted percentage strings.
    """
    # Round the values and convert them to percentage strings
    data_percent = [f"{value * 100:.{decimal_places}f}%" for value in data]

    # Create a DataFrame for the updated posterior percentages
    posterior_df = pd.DataFrame(
        [data_percent],    # Percentage values as a list
        columns=column_names         # Column labels for the DataFrame
    )

    # Display the DataFrame
    print(header_text)
    # In environments like scripts, returning the DataFrame is more standard
    # than relying on implicit display.
    return posterior_df

def setup_logging(run_directory: Path):
    """Configures the application's root logger for the parallel runner.

    This function sets up a clean, dual-output logging system for the main
    execution script. A critical action it performs is to **clear any
    existing handlers** attached to the root logger. This is essential to
    prevent duplicate log output that can occur in environments where the
    logging system may have been previously configured (e.g., in a Jupyter notebook).

    After clearing, it adds two new handlers:
    1.  A `StreamHandler` to direct logs to the console (`sys.stdout`).
    2.  A `FileHandler` to save logs to a file named `parallel_runner.log`
        within the specified `run_directory`.

    Args:
        run_directory: The unique top-level directory for the entire
            parallel run, where the `parallel_runner.log` file will be saved.

    Side Effects:
        - Modifies the global root logger (`logging.getLogger()`) in-place.
        - Clears all pre-existing handlers from the root logger.
        - Creates the `run_directory` on the filesystem if it does not exist.
        - Creates and writes to the `parallel_runner.log` file.
    """
    # Define the log file path.
    log_file = run_directory / "experiment.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Get the root logger.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set the minimum level of messages to handle.

    # Clear any existing handlers to avoid duplicate logs.
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a formatter to define the log message format.
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create a handler to write to the console (stdout).
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create a handler to write to the log file.
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)