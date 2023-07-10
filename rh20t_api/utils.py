"""
    Utilities.
"""

import json
import numpy as np

def load_json(path:str):
    with open(path, "r") as _json_file: _json_content = json.load(_json_file)
    return _json_content

def write_json(path:str, _json_content):
    with open(path, "w") as _json_file: json.dump(_json_content, _json_file)
    
def load_dict_npy(file_name:str): 
    """
        Load the dictionary data stored in .npy file.
    """
    return np.load(file_name, allow_pickle=True).item()