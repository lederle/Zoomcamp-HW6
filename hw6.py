from pathlib import Path
import pandas as pd
import urllib.request

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from tqdm.auto import tqdm
import xgboost as xgb

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

def random_forest_regression(rf, X, y, val, y_val, Q2 = True):
    rf.fit(X, y)
    pred = rf.predict(val)
    rmse = np.sqrt(mean_squared_error(pred, y_val))
    if Q2:
        print(rmse)
    return rmse 

def optimize_N_estimators(X, y, val, y_val):
    scores = []
    for n in tqdm(range(10, 201, 10)):
        rf = RandomForestRegressor(n_estimators = n, random_state = 1, n_jobs = -1)
        scores.append(random_forest_regression(rf, X, y, val, y_val, Q2 = False))
    return pd.DataFrame({"n_estimator": list(range(10, 201, 10)), "scores": scores})
        
def optimize_max_depth(X, y, val, y_val):
    scores = []

    for d in tqdm([10, 15, 20, 25]):
        rf = RandomForestRegressor(n_estimators = 1, max_depth = d, random_state = 1, warm_start = True)
        for n in tqdm(range(10, 201, 10)):
            rf.n_estimators = n
            scores.append((d, n, random_forest_regression(rf, X, y, val, y_val, Q2 = False)))
    columns = ["max_depth", "n_estimator", "rmse"]
    return pd.DataFrame(scores, columns = columns)

def xgb_compare_eta(dv, X, y, val, y_val):
    features = dv.get_feature_names_out()
    dtrain = xgb.DMatrix(X, label = y, feature_names = features)
    dval = xgb.DMatrix(val, label = y_val, feature_names = features)
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    scores = {}
    for eta in [0.3, 0.1]:
        xgb_params = {
            'eta': eta,
            'max_depth': 6,
            'min_child_weight': 1,
            'objective': 'reg:squarederror',
            'nthread': 8,
            'seed': 1,
            'verbosity': 1,
        }

        progress = {}
        model = xgb.train(xgb_params, dtrain, num_boost_round=100,
                          verbose_eval=5, evals=watchlist, evals_result = progress)

        idx = f"eta={str(eta)}"
        scores[idx] = pd.DataFrame({"round": list(range(100)),
                        "train": progress["train"]["rmse"],
                        "val": progress["val"]["rmse"]})
    return scores
