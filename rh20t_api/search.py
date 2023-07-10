"""
    Searching functions for interpolations
"""

import numpy as np
from math import *

def binary_search_latest_range(arr, l:int, r:int, x):
    if arr[r] <= x or r == l: return arr[r]
    mid = ((l + r) >> 1) + 1
    return binary_search_latest_range(arr, mid, r, x) if arr[mid] <= x else binary_search_latest_range(arr, l, mid - 1, x)

def binary_search_latest(arr:list, x):
    '''
        search for the nearest item in arr just smaller than x,
        if no one smaller than x is found, return the smallest
        
        Params:
        ----------
        arr:    the array to search on
        x:      the target value
        
        Returns:
        ----------
        x_t:    the closest previous value in arr of x
    '''
    if len(arr) <= 0: raise ValueError("input array should contain at least one element")
    return binary_search_latest_range(arr, 0, len(arr) - 1, x)

def interpolate_linear(target_t:int, t1:int, t2:int, x1:np.ndarray, x2:np.ndarray): return (x1 + (target_t - t1) / (t2 - t1) * (x2 - x1) if t1 != t2 else x1)

def binary_search_closest_two_idx(t:list, target_t:int):
    # linearly searches indices of two closest element of target_t in t
    # for continuous values (i.e. non-image data) searching
    prev_t_idx = t.index(binary_search_latest(t, target_t))
    return (prev_t_idx, (prev_t_idx + 1 if prev_t_idx < len(t) - 1 else prev_t_idx - 1))

def binary_search_closest(t:list, target_t:int):
    # for image path searching
    if target_t in t: return target_t
    prev_t_idx = t.index(binary_search_latest(t, target_t))
    if prev_t_idx == len(t) - 1: return t[prev_t_idx]
    return t[prev_t_idx] if abs(t[prev_t_idx] - target_t) < abs(t[prev_t_idx + 1] - target_t) else t[prev_t_idx + 1]
    
def sort_by_timestamp(_dict_of_list_of_dict):
    for _k in _dict_of_list_of_dict: _dict_of_list_of_dict[_k] = sorted(_dict_of_list_of_dict[_k], key=lambda item:item["timestamp"])