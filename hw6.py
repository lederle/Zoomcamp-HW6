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

IMAGES_PATH = Path() / "images"
IMAGES_PATH.mkdir(parents = True, exist_ok = True)

def save_fig(fig_id, tight_layout = True, fig_extension = "png", resolution = 300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
        plt.savefig(path, format = fig_extension, dpi = resolution)

