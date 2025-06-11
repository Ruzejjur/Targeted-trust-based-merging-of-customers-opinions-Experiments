
# Matplotlib: Plotting library
import matplotlib.pyplot as plt

# os: Interact with the operating system
import os

# Pandas: Data manipulation and analysis
import pandas as pd

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