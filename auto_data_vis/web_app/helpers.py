import json
from datetime import datetime
import numpy as np
from collections import Counter
from scipy.stats import entropy
import pandas as pd
from scipy.stats import pearsonr

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def filter_data(data):
    #remove special characters and empty or None entries
    data = [item.replace('\xa0', '').replace('\n', '').replace('(mL)', '') for item in data if isinstance(item, str)]
    data = [item for item in data if item]
    return data

def is_wrong(values):
    try:
        # Check if values is a list of lists
        if isinstance(values, list) and all(isinstance(element, list) for element in values):
            return True
        if not values:
            return True
        if all(element == "" for element in values):
            return True
        if len(values)==0:
            return True
        if all(element == values[0] for element in values):
            return True
    except TypeError:
        return True
    
def calculate_overlap(a_data, b_data):
    a_min, a_max = np.min(a_data), np.max(a_data)
    a_range = a_max - a_min
    b_min, b_max = np.min(b_data), np.max(b_data)
    b_range = b_max - b_min
    has_overlap = False
    overlap_percent = 0
    if (a_max >= b_min) and (b_min >= a_min):
        has_overlap = True
        overlap = (a_max - b_min)
    if (b_max >= a_min) and (a_min >= b_min):
        has_overlap = True
        overlap = (b_max - a_min)
    if has_overlap:
        overlap_percent = max(overlap / a_range, overlap / b_range)
    if ((b_max >= a_max) and (b_min <= a_min)) or (
            (a_max >= b_max) and (a_min <= b_min)):
        has_overlap = True
        overlap_percent = 1
    return has_overlap, overlap_percent

def identical_elements(a, b):
    num_identical = 0
    
    # Determine the length of the shorter list
    min_length = min(len(a), len(b))
    
    # Iterate through the corresponding elements of both lists up to the length of the shorter list
    for i in range(min_length):
        if a[i] == b[i]:
            num_identical += 1
    
    return num_identical

def is_numerical_string(value):
    cleaned_value = value.replace('$', '').replace('%', '')
    cleaned_value = cleaned_value.replace(',', '')
    return cleaned_value.replace('.', '', 1).isdigit()

def count_non_numerical_elements(filtered_data):
    count = 0
    for value in filtered_data:
        if not is_numerical_string(value):
            count += 1
    return count

def convert_to_numeric_list(lst):
    cleaned_lst = []
    for item in lst:
        if item is not None:  # Exclude null values
            if isinstance(item, str):
                # Remove unwanted characters
                item = item.replace('%', '').replace('$', '').replace('(mL)', '').replace('nan','').replace('NaN','')
                if ',' in item:
                    item = item.replace(',', '')

                # Handle negative numbers properly
                if '-' in item and 'E' not in item:
                    item = '-' + item.replace('-', '')

            # Attempt to convert to float, and if successful, add to cleaned list
            try:
                cleaned_item = float(item)
                cleaned_lst.append(cleaned_item)
            except ValueError:
                pass  # Skip the problematic element and continue

    return cleaned_lst

def is_numerical(lst):
    for item in lst:
        if not isinstance(item, (int, float)):
            return False
    return True

def flatten_list(nested_list):
    flattened_list = []
    for sublist in nested_list:
        if isinstance(sublist, list):
            flattened_list.extend(sublist)
        else:
            flattened_list.append(sublist)
    return flattened_list

def check_dates(date_list):
    try:
        parsed_dates = [datetime.strptime(date, "%m/%d/%y") for date in date_list]
        return 'd'
    except ValueError:
        return 'u'
    
def is_number(value):
    symbols = ['$','€','£','¥','%']
    return any(symbol in value and any(char.isdigit() for char in value) and not any(char.isalpha() for char in value) for symbol in symbols)

def is_string(element):
    if isinstance(element, str):
        # Check if the string contains any alphabetic characters
        return any(char.isalpha() for char in element)
    return False

def check_dates(date_list):
    try:
        parsed_dates = [datetime.strptime(date, "%m/%d/%y") for date in date_list]
        return 'd'
    except ValueError:
        return 'u'
    
def list_entropy(l):
    return entropy(pd.Series(l).value_counts() / len(l))

def insert_missing_values(list_x, list_y,type_x,type_y):
    len_x = len(list_x)
    len_y = len(list_y)
    if len_x == len_y:
        return list_x, list_y
    if len_x < len_y:
        smaller_list = list_x
        smaller_list_type = type_x
        larger_list = list_y
    else:
        smaller_list = list_y
        smaller_list_type = type_y
        larger_list = list_x

    num_missing_values = abs(len_x - len_y)

    if smaller_list_type in ['c','d']:
        most_common_label = Counter(smaller_list).most_common(1)[0][0]
        smaller_list_filled = smaller_list + [most_common_label] * num_missing_values
    if smaller_list_type == 'q':
        # insert median
        median = np.median(smaller_list)
        smaller_list_filled = smaller_list + [median] * num_missing_values

    if len_x < len_y:
        return smaller_list_filled, larger_list
    else:
        return larger_list, smaller_list_filled

def calc_entropy(values):
    positive_values = [v for v in values if v >= 0]
    ent = entropy(positive_values)
    if ent == 0 or len(positive_values) == 0:
        positive_values = [abs(v) for v in values if v < 0]
    return entropy(positive_values)

def calculate_sortedness(values, sorted_v):
    return np.absolute(pearsonr(values, sorted_v)[0])