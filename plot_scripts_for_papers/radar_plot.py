import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def radar_chart(df):
    """
    Plots a radar chart for content of dataframe. Expects input in the form of dataframe with categories as columns and
    cases to be plotted as indices. Names on plot taken from index and column names.
    """
    # Number of variables
    num_vars = df.shape[1]

    # Compute angle each bar is centered on:
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Make plot close to a circle
    values = df.iloc[0].values.flatten().tolist()
    values += values[:1]
    angles += angles[:1]

    # Initialise the radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], df.columns, color='grey', size=10)

    # Draw ylabels
    r_max = df.to_numpy().max()
    plt.yticks(np.arange(0, r_max + r_max / 10, r_max / 10), color="grey", size=7)
    plt.ylim(0, r_max)

    # Plot data for each row in the DataFrame
    for idx, row in df.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1]

        # Plot data
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=idx)

        # Fill area
        ax.fill(angles, values, alpha=0.1)

    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)

    ax.legend()

    return fig

if __name__ == '__main__':
    plt.style.use('ml_colors')
    # Example usage:
    data = {
        'Project 1': [5, 6, 6, 8, 4],
        'Project 2': [2, 8, 7, 4, 4],
        'Project 3': [5, 4, 5, 6, 5],
        'Project 4': [10, 5, 7, 8, 8]
    }

    df = pd.DataFrame.from_dict(data, orient='index')
    df.columns = ['Resource Need', 'Academic Level', 'Potential Value', 'Partner Interest', 'Risk of Failure']

    fig = radar_chart(df)
