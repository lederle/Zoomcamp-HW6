from pathlib import Path
import pandas as pd
import urllib.request

def load_housing_data():
    csv_path = Path("datasets/housing.csv")
    if not csv_path.is_file():
        Path("datasets").mkdir(parents = True, exist_ok = True)
        url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv"
        urllib.request.urlretrieve(url, csv_path)
    return pd.read_csv(Path("datasets/housing.csv"))
