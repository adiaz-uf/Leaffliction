import os
import sys
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

# Data path
data_path = './data/original/'


def extract_data(subdir) -> pd.DataFrame:
    # Concatenate Data Path and recieved dir
    file_path = data_path + subdir

    base_dir = pathlib.Path(file_path)
    directory_name = base_dir.name

    # 1. Extract data
    data = {}
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            # Count images in each subdir
            count = len(list(subdir.glob('*')))
            data[subdir.name] = count
    return data, directory_name


def plot_data(dir_to_plot):
    data, plant_type = extract_data(dir_to_plot)

    # 2. Analyze and graph data
    if data:
        # Convert to pandas DataFrame
        df = pd.DataFrame.from_dict(data, orient='index', columns=['count'])
        print(df, '\n')

        # Create figure with 2 axes to insert 2 graphs
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

        # Get a list of default matplotlib colors
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors_list = default_colors * (len(df) // len(default_colors) + 1)
        colors = colors_list[:len(df)]

        # Pie Chart
        df.plot(kind='pie',
                title=f'{plant_type} class distribution',
                y='count',
                autopct='%1.1f%%',
                legend=False,
                ylabel='',
                ax=axes[0],
                labels=None,
                colors=colors)

        # Bar Chart
        bar_chart = df['count'].plot(kind='bar',
                                     legend=False,
                                     ax=axes[1],
                                     color=colors)

        bar_chart.tick_params(axis='x', rotation=0)

        # Adjust design
        plt.tight_layout()

        plt.show()


def main():
    if len(sys.argv) < 2:
        print('Error! Usage is: ./Distribution.py <subdir>')
        sys.exit(1)

    subdir = sys.argv[1]

    if not os.path.isdir(data_path + subdir):
        print('Error! Invalid directory')
        sys.exit(2)

    plot_data(subdir)


if __name__ == '__main__':
    main()
