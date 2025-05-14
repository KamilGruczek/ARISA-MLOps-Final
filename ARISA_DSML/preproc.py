from loguru import logger
from pathlib import Path
import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from ARISA_DSML.config import DATASET, PROCESSED_DATA_DIR, RAW_DATA_DIR


def get_raw_data(dataset:str=DATASET)->None:
    api = KaggleApi()
    api.authenticate()

    download_folder = Path(RAW_DATA_DIR)
    logger.info(f"RAW_DATA_DIR is: {RAW_DATA_DIR}")
    api.dataset_download_files(dataset, path=str(download_folder), unzip=True)


def preprocess_df(file:str|Path)->str|Path:
    """Preprocess dataset."""
    _, file_name = os.path.split(file)
    df_data = pd.read_csv(file)

    # Split date to year, month
    df_data['Time'] = pd.to_datetime(df_data['Time'])
    df_data['Year'] = df_data['Time'].dt.year
    df_data['Month'] = df_data['Time'].dt.month

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    outfile_path = PROCESSED_DATA_DIR / file_name
    df_data.to_csv(outfile_path, index=False)

    return outfile_path


if __name__=="__main__":
    # get the train and test sets from default location
    logger.info("getting datasets")
    get_raw_data()

    # preprocess Location1 set
    logger.info("Location1.csv")
    preprocess_df(RAW_DATA_DIR / "Location1.csv")
