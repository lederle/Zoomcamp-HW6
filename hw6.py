from pathlib import Path
import pandas as pd
import urllib.request

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import export_text

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

def fill_na(df):
    df.fillna(0, inplace = True)

def transform_label(df):
   return np.log1p(df["median_house_value"])

def plot_log_transform_of_median_house_price(df):
    fig, axs = plt.subplots(1, 2, figsize = (8, 3), sharey = True)
    df["median_house_value"].hist(ax = axs[0], bins = 50)
    df["median_house_value"].apply(np.log).hist(ax = axs[1], bins = 50)
    axs[0].set_ylabel("Number of districts")
    axs[0].set_xlabel("Median House Value")
    axs[1].set_xlabel("Log of Median House Value")
    save_fig("log_transform_median_house_price")
    plt.show()

def prepare_data(df):
    full, test = train_test_split(df, test_size = 0.2, random_state = 1)
    train, val = train_test_split(full, test_size = 0.25, random_state = 1)
    for d in [train, val, test]:
        d.reset_index(drop = True, inplace = True)

    y_train = train["median_house_value"].values
    y_val = val["median_house_value"].values
    y_test = test["median_house_value"].values

    del train["median_house_value"]
    del val["median_house_value"]
    del test["median_house_value"]

    return (train, y_train), (val, y_val), (test, y_test)

def dict_vectorize(dv, df):
    return dv.transform(rows_as_dict(df))

def rows_as_dict(data):
    return data.to_dict(orient = 'records')

def tree_regression(dtr, dv, X, y):
    dtr.fit(X, y)
    print(export_text(dtr, feature_names = dv.get_feature_names_out().tolist()))
