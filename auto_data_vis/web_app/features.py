from helpers import *
import pandas as pd
from scipy.stats import entropy, normaltest, mode, kurtosis, skew, pearsonr, moment
from scipy.stats import pearsonr, f_oneway, chi2_contingency, ks_2samp
from datetime import *

def uniqueness_features(values):
    values = [value for value in values if value != ""]
    values1 = []
    for item in values:
        if item is not None:
            if isinstance(item, str):
                item = item.replace('\xa0', ' ').replace('\n', '').replace('$', '').replace('(mL)', "")
            values1.append(item)

    num_unique = len(set(flatten_list(values1)))
    num_entries = len(values1)

    num_none = sum(1 for e in values1 if (pd.isnull(e) or e ==''))
    percentage_none = num_none / num_entries
    has_none = (num_none > 0)
    return values1, num_unique, num_entries, num_none, percentage_none,has_none

# general pairwise features
def g_pairwise_features(a,b):
    a = np.array(a)
    b = np.array(b)   
    num_identical_elements = identical_elements(a,b)
    has_shared_elements = (num_identical_elements > 0)
    percent_shared_elements = num_identical_elements / len(a)
    identical = num_identical_elements / len(a)

    a_unique_data = set(a)
    b_unique_data = set(b)
    num_shared_unique_elements = len(a_unique_data.intersection(b_unique_data))
    has_shared_unique_elements = (num_shared_unique_elements > 0)
    percent_shared_unique_elements = num_shared_unique_elements/ max(len(a_unique_data), len(b_unique_data))
    identical_unique = (a_unique_data == b_unique_data) 

    return num_identical_elements, has_shared_elements, percent_shared_elements, identical, num_shared_unique_elements, has_shared_unique_elements, percent_shared_unique_elements, identical_unique

# statistical pairwise features
def s_pairwise_features(a,b,type_a, type_b):
    a = np.array(a)
    b = np.array(b)  
    min_len = min(len(a), len(b))
    a_data = a[:min_len]
    b_data = b[:min_len] 
    if (type_a == 'q' and type_b == 'q'):
        correlation_value, correlation_p = pearsonr(a_data, b_data)
        ks_statistic, ks_p = ks_2samp(a_data, b_data)
        #has_overlap, overlap_percent = calculate_overlap(a_data, b_data)
        correlation_significant_005 = (correlation_p < 0.05) 
        ks_significant_005 = (ks_p < 0.05)
    else:
        correlation_significant_005 = ks_significant_005 = False
        correlation_value = correlation_p = 0
        ks_statistic = ks_p = 0
        #has_overlap = overlap_percent = 0
    return correlation_value, correlation_p, ks_statistic, ks_p, correlation_significant_005, ks_significant_005

def categorical_features(values, value_lengths):
    c_entropy = list_entropy(values)
    mean_c_length = np.mean(value_lengths)
    median_c_length = np.median(value_lengths)  
    min_c_length = np.min(value_lengths)
    max_c_length = np.max(value_lengths)
    std_c_length = np.std(value_lengths)
    percentage_mode_c = (pd.Series(values).value_counts().max() / len(values))

    unique_categories = len(set(values))


    return c_entropy, mean_c_length, median_c_length, min_c_length, max_c_length, std_c_length, percentage_mode_c,unique_categories
    
def quantitive_features(values):
    sample_mean = np.mean(values)
    sample_median = np.median(values)
    sample_var = np.var(values)
    sample_min = np.min(values)
    sample_max = np.max(values)
    sample_std = np.std(values)
    if sample_max != 0:
        normalized_mean = sample_mean / sample_max
        normalized_median = sample_median / sample_max
    else:
        normalized_mean = 0
        normalized_median = 0
    coeff_var = (sample_mean / sample_var) if sample_var else None
    range = sample_max - sample_min
    normalized_range = (sample_max - sample_min) / \
        sample_mean if sample_mean else None
    
    q_entropy = calc_entropy(values)
    kurt = kurtosis(values)
    skewness = skew(values)

    #outliers
    q1, q25, q75, q99 = np.percentile(values, [0.01, 0.25, 0.75, 0.99])
    iqr = q75 - q25
    outliers_15iqr = np.logical_or(
            values < (q25 - 1.5 * iqr), values > (q75 + 1.5 * iqr))
    outliers_3iqr = np.logical_or(values < (q25 - 3 * iqr), values > (q75 + 3 * iqr))
    outliers_1_99 = np.logical_or(values < q1, values > q99)
    outliers_3std = np.logical_or(
            values < (
                sample_mean -
                3 *
                sample_std),
            values > (
                sample_mean +
                3 *
                sample_std))
    percent_outliers_15iqr = np.sum(outliers_15iqr) / len(values)
    percent_outliers_3iqr = np.sum(outliers_3iqr) / len(values)
    percent_outliers_1_99 = np.sum(outliers_15iqr) / len(values)
    percent_outliers_3std = np.sum(outliers_15iqr) / len(values)

    has_outliers_15iqr = np.any(outliers_15iqr)
    has_outliers_3iqr = np.any(outliers_3iqr)
    has_outliers_1_99 = np.any(outliers_1_99)
    has_outliers_3std = np.any(outliers_3std)

    return (sample_mean, sample_median, sample_var, sample_min, sample_max, 
            sample_std, normalized_mean, normalized_median, coeff_var, 
            range, normalized_range, q_entropy, kurt, skewness,
            percent_outliers_15iqr ,percent_outliers_3iqr,
            percent_outliers_1_99, percent_outliers_3std,
            has_outliers_15iqr, has_outliers_3iqr, has_outliers_1_99, has_outliers_3std)

def sequence_features(values,type):
    sorted_v = np.sort(values)
    if type == 'd':
        sortedness = None
        is_sorted = None
    if type == 'c' :
        sortedness = None
        is_sorted = np.array_equal(sorted_v, values)
    if type == 'q':
        sortedness = calculate_sortedness(values, sorted_v)
        is_sorted = np.array_equal(sorted_v, values)
    return sortedness, is_sorted
        

        