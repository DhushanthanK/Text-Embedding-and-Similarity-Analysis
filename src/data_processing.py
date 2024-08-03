# src/data_processing.py

import os
import pandas as pd

def load_data(file_path):
    if os.path.exists(file_path):
        print("File exists at the specified path.")
    else:
        print("File does not exist at the specified path.")
    data = pd.read_csv(file_path, sep='\t')
    return data
