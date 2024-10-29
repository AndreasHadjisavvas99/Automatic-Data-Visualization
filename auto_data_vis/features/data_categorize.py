import os 
import json
import numpy as np
import pandas as pd
from datetime import *
from sklearn.utils import shuffle
from collections import OrderedDict
from scipy.stats import entropy, normaltest, mode, kurtosis, skew, pearsonr, moment
from scipy.stats import pearsonr, f_oneway, chi2_contingency, ks_2samp
from helpers import *
from data_categorize import *
from features import *

# categorize data into datetime 'd' or quantitative 'q' or unknown 'x'
def categorize(data,file_path):
    filtered_data = filter_data(data)

    if len(data) < 3:
        return 'x'
    try:
        if is_string(data[0]) and not is_string(data[1]):
            temp1 = data[0]
            filtered_data = [str(x).strip() for x in data[1:] if isinstance(x, str) and str(x).strip() != "" and str(x).strip() != "-"]
        if is_string(data[0]) and is_string(data[1]) and not is_string(data[2]):
            temp1 = data[0]
            temp2 = data[1]
            filtered_data = [str(x).strip() for x in data[2:] if isinstance(x, str) and str(x).strip() != "" and str(x).strip() != "-"]
    except:
        print("exception", data)

    #attempt to convert to float
    copied_data = filtered_data[:]
    conversion_failed = False
    for i, x in enumerate(filtered_data):
        x_no_commas = x.replace(',', '')
        try:
            float_value = float(x_no_commas)
            filtered_data[i] = float_value
        except ValueError:
            conversion_failed = True
            break
    if conversion_failed == False:
        
        return 'q'
    else:
        filtered_data = copied_data
    #print(filtered_data)
    # Check if all elements are numeric or formatted as money
    if all((isinstance(x, (int, float)) or (isinstance(x, str) and is_number(x))) for x in filtered_data):
        return 'q'
    if check_dates(filtered_data) == 'd':
        return 'd'
    if is_numerical(filtered_data):
        return 'q'
    # Check if all elements match the datetime format 'yyyy-m-d' or 'yyyy-mm-d' or 'yyyy-m-dd' or 'yyyy-mm-dd'
    if all(isinstance(x, str) and (len(x) == 10 or len(x) == 9 or len(x) == 8) and x[4] == '-' and (x[7] == '-' or x[6] == '-') and all(c.isdigit() for c in x[:4] + x[5]) for x in filtered_data):
        return 'd'
    # Check if all elements match the datetime format 'YYYY-YY-yy xx:xx:xx'
    if all(isinstance(x, str) and (len(x) == 10 or len(x) == 19) and x[4] == '-' and x[7] == '-'and all(c.isdigit() for c in x[:4] + x[5:7]) for x in filtered_data):
        return 'd'
    # Check if all elements match the datetime format 'YYYY-YY'
    if all(isinstance(x, str) and (len(x) == 7 or len(x) == 6) and x[4] == '-' and all(c.isdigit() for c in x[:4] + x[5:]) for x in filtered_data):
        return 'd'
    # Check if all elements match the datetime format 'YYYY-YYYY'
    if all(isinstance(x, str) and len(x) == 9 and x[4] == '-' and all(c.isdigit() for c in x[:4] + x[5:]) for x in filtered_data):
        return 'd'
    # Check if all elements match the datetime format 'YYYY-MM-DD'
    if all(isinstance(x, str) and len(x) == 10 and x[4] == '-' and x[7] == '-' and all(c.isdigit() for c in x[:4] + x[5:7] + x[8:]) for x in filtered_data):
        return 'd'
    # Check if all elements match the datetime format 'MM/DD/YYYY'
    if all(isinstance(x, str) and len(x) == 10 and x[2] == '/' and x[5] == '/' and all(c.isdigit() for c in x[:2] + x[3:5] + x[6:]) for x in filtered_data):
        return 'd'
    # Check if all elements match the datetime format 'MM/DD/YY'
    if all(isinstance(x, str) and len(x) == 8 and (x[2] == '/' or x[2] == '.') and (x[5] == '/' or x[5] == '.') and all(c.isdigit() for c in x[:2] + x[3:5] + x[6:]) for x in filtered_data):
        return 'd'
    # Check if all elements match the datetime format 'M/D/YYYY'
    if all(isinstance(x, str) and (len(x) == 9 or len(x) == 10) and x[1] == '/' and x[3] == '/' and all(c.isdigit() for c in x[:1] + x[2:3] + x[4:]) for x in filtered_data):
        return 'd'
    # Check if all elements match the datetime format "1/1/2010" or "11/26/2010"
    if all(isinstance(x, str) and len(x) <= 10 and len(x) >= 8 and x.count('/') >= 2 and all(c.isdigit() for c in x if c != '/') for x in filtered_data):
        return 'd'
    if all(isinstance(x, str) and len(x) == 7 and x[4] == '/' and all(c.isdigit() for c in x[:4] + x[5:]) for x in filtered_data):
        return 'd'
    # Check if all elements match the datetime format "*/*"
    if all(isinstance(x, str) and len(x) <= 5 and len(x) >= 3 and x.count('/') ==1  and all(c.isdigit() for c in x if c != '/') for x in filtered_data):
        return 'd'
    if all(isinstance(x, str) for x in filtered_data):
        # check if strings falsely exist in numerical lists
        #num_strings = sum(1 for value in filtered_data if not value.replace(',', '').replace('$', '').replace('.', '').isdigit())
        num_strings = count_non_numerical_elements(filtered_data)
        num_numerical = len(filtered_data) - num_strings
        string_ratio = num_strings / len(filtered_data)
        numerical_ratio = num_numerical / len(filtered_data)
        if string_ratio < 0.1 * numerical_ratio:
            return 'q'
        return 'c'
    if all(isinstance(x, str) and x.lower() in {'true', 'false'} for x in filtered_data):
        return 'b'
    return 'u'