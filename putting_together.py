"""The content of this file should be appended to the preprocessing
"""
# import os
import pickle

import pandas as pd

from constants import PROCESSED_PATH

if __name__ == "__main__":
    dataframes = []
    with open(PROCESSED_PATH / "paths.pickle", "rb") as handle:
        paths = pickle.load(handle)
    for path in paths:
        curr_dataframe = pd.read_csv(path)
        dataframes.append(curr_dataframe)
    concat_dataframe = pd.concat(dataframes)
    concat_dataframe.to_csv(PROCESSED_PATH / "10m_dataframe.csv")
    # for path in paths:
    #    os.remove(path)
