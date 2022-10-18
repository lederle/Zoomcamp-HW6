from pathlib import Path
import pandas as pd
import urllib.request

import matplotlib.pyplot as plt
import numpy as np

def load_housing_data():
    csv_path = Path("datasets/housing.csv")
    if not csv_path.is_file():
        Path("datasets").mkdir(parents = True, exist_ok = True)
        url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv"
        urllib.request.urlretrieve(url, csv_path)
    return pd.read_csv(Path("datasets/housing.csv"))

IMAGES_PATH = Path() / "images"
IMAGES_PATH.mkdir(parents = True, exist_ok = True)

def save_fig(fig_id, tight_layout = True, fig_extension = "png", resolution = 300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
        plt.savefig(path, format = fig_extension, dpi = resolution)

def plot_log_transform_of_median_house_price(df):
    fig, axs = plt.subplots(1, 2, figsize = (8, 3), sharey = True)
    df["median_house_value"].hist(ax = axs[0], bins = 50)
    df["median_house_value"].apply(np.log).hist(ax = axs[1], bins = 50)
    axs[0].set_ylabel("Number of districts")
    axs[0].set_xlabel("Median House Value")
    axs[1].set_xlabel("Log of Median House Value")
    save_fig("log_transform_median_house_price")
    plt.show()

