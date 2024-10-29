from datetime import *
from helpers import *
from data_categorize import *
from features import *
import sys 
from features import uniqueness_features, categorical_features, quantitive_features, sequence_features, s_pairwise_features, g_pairwise_features

def calc_features_json(f):
    f_data_entry = f['data'][0]
    plot_type = f_data_entry.get('type')
    if plot_type == 'scatter' and 'mode' in f_data_entry and f_data_entry['mode'] == 'lines':
        plot_type = 'line'
    try:
        if plot_type == 'pie':
            x_values = f_data_entry['labels']
            y_values = f_data_entry['values']
        elif plot_type == 'histogram':
            x_values = f_data_entry['x']
            y_values = []
        else:
            x_values = f_data_entry['x']
            y_values = f_data_entry['y']
    except Exception as e:
        print("Exception found:",e)
    if is_wrong(x_values):
        print("Found error in x values")
    if plot_type != 'histogram':
        if is_wrong(y_values):
            print("Found error in y values")
    #type
    data_type_x = categorize(x_values)
    if plot_type != 'histogram':
        data_type_y = categorize(y_values)
    else:
        data_type_y = None
    if (data_type_x == 'x' or data_type_y == 'x'):
        print("Found error in data type calculation")
    
    #convert to numeric
    if (data_type_x == 'q'):
        try:
            x_values = convert_to_numeric_list(x_values)
        except Exception as e:
            print(e)

        if is_wrong(x_values):
            print("Found error in x values, line 57")
    if (data_type_y == 'q'):
        y_values = convert_to_numeric_list(y_values)
        leny = len(y_values)
        if is_wrong(y_values):
            print("Found error in x values, line 62")
    
    #uniqueness  features
    x_values, num_unique_x, num_entries_x, num_none_x, percentage_none_x, has_none_x = uniqueness_features(x_values)
    y_values, num_unique_y, num_entries_y, num_none_y, percentage_none_y, has_none_y = uniqueness_features(y_values)
    
    # discard data with unequal number of entries
    entry_difference = abs(num_entries_x - num_entries_y)
    max_entries = max(num_entries_x, num_entries_y)
    percentage_difference = entry_difference / max_entries
    if percentage_difference > 0.2:
        print("Error Unequal number of entries")
    if num_entries_x != num_entries_y:
        x_values , y_values = insert_missing_values(x_values, y_values, data_type_x, data_type_y)
    if (len(x_values) != len(y_values)):
        print(num_entries_x, ' ',num_entries_y)

    # general pairwise features
    num_identical_elements, has_shared_elements, percent_shared_elements, identical, num_shared_unique_elements, has_shared_unique_elements, percent_shared_unique_elements, identical_unique = g_pairwise_features(x_values,y_values)

    # statistical pairwise features
    correlation_value, correlation_p, ks_statistic, ks_p, correlation_significant_005, ks_significant_005 = s_pairwise_features(x_values,y_values,data_type_x,data_type_y)
    # has_overlap, overlap_percent

    #categorical features
    if data_type_x in ('c', 'd'):
        try:
            value_lengths_x = [len(x) for x in x_values]
            c_entropy_x, mean_c_length_x, median_c_length_x, min_c_length_x, max_c_length_x, std_c_length_x, percentage_mode_c_x, unique_categories_x = categorical_features(x_values,value_lengths_x)
        except Exception as e:
            print("Error in calculating categorical features for x")
            print("Found exception:",e)
    else:
        c_entropy_x, mean_c_length_x, median_c_length_x, min_c_length_x, max_c_length_x, std_c_length_x, percentage_mode_c_x,unique_categories_x = None, None, None, None, None, None, None, None

    if data_type_y in ('c', 'd'):
        try:
            value_lengths_y = [len(x) for x in y_values]
            c_entropy_y, mean_c_length_y, median_c_length_y, min_c_length_y, max_c_length_y, std_c_length_y, percentage_mode_c_y, unique_categories_y = categorical_features(y_values,value_lengths_y)
        except Exception as e:
            print("Error in calculating categorical features for y")
            print("Found exception:",e)
    else:
        c_entropy_y, mean_c_length_y, median_c_length_y, min_c_length_y, max_c_length_y, std_c_length_y, percentage_mode_c_y, unique_categories_y= None, None, None, None, None, None, None, None
        
    # quantitive features
    if (data_type_x == 'q'):
        sample_mean_x, sample_median_x, sample_var_x, sample_min_x, sample_max_x, sample_std_x, normalized_mean_x, normalized_median_x, coeff_var_x, range_x, normalized_range_x, q_entropy_x, kurt_x,skewness_x, percent_outliers_15iqr_x ,percent_outliers_3iqr_x, percent_outliers_1_99_x, percent_outliers_3std_x, has_outliers_15iqr_x, has_outliers_3iqr_x, has_outliers_1_99_x, has_outliers_3std_x = quantitive_features(x_values)
    else:
        sample_mean_x, sample_median_x, sample_var_x, sample_min_x, sample_max_x, sample_std_x = None, None, None, None, None, None
        normalized_mean_x, normalized_median_x, coeff_var_x, range_x, normalized_range_x, q_entropy_x = None, None, None, None, None, None
        kurt_x,skewness_x = None, None
        percent_outliers_15iqr_x ,percent_outliers_3iqr_x, percent_outliers_1_99_x, percent_outliers_3std_x, has_outliers_15iqr_x, has_outliers_3iqr_x, has_outliers_1_99_x, has_outliers_3std_x = None, None, None, None, None, None, None, None

    if (data_type_y == 'q'):
        sample_mean_y, sample_median_y, sample_var_y, sample_min_y, sample_max_y, sample_std_y, normalized_mean_y, normalized_median_y, coeff_var_y, range_y, normalized_range_y, q_entropy_y, kurt_y,skewness_y, percent_outliers_15iqr_y ,percent_outliers_3iqr_y, percent_outliers_1_99_y, percent_outliers_3std_y, has_outliers_15iqr_y, has_outliers_3iqr_y, has_outliers_1_99_y, has_outliers_3std_y = quantitive_features(y_values)
    else:
        sample_mean_y, sample_median_y, sample_var_y, sample_min_y, sample_max_y, sample_std_y = None, None, None, None, None, None
        normalized_mean_y, normalized_median_y, coeff_var_y, range_y, normalized_range_y, q_entropy_y = None, None, None, None, None, None
        kurt_y,skewness_y = None, None
        percent_outliers_15iqr_y ,percent_outliers_3iqr_y, percent_outliers_1_99_y, percent_outliers_3std_y, has_outliers_15iqr_y, has_outliers_3iqr_y, has_outliers_1_99_y, has_outliers_3std_y = None, None, None, None, None, None, None, None
    
    # sequence features
    try:
        sortedness_x, is_sorted_x = sequence_features(x_values, data_type_x)
    except Exception as e:
        print("Error found in sortedness features for x: ")
        print(e)
        sys.exit(1)
    #sortedness_y, is_sorted_y = sequence_features(y_values, data_type_y)

    try:
        sortedness_y, is_sorted_y = sequence_features(y_values, data_type_y)
    except Exception as e:
        print("Error found in sortedness features for y: ")
        print(e)
        sys.exit(1)

    values = [data_type_x, data_type_y, num_entries_x, num_entries_y,
                            normalized_range_x, normalized_range_y,
                            num_unique_x, num_unique_y,
                            has_none_x, num_none_x,percentage_none_x,
                            has_none_y, num_none_y,percentage_none_y,
                            normalized_mean_x, normalized_median_x, sample_var_x, 
                            sample_std_x, coeff_var_x, sample_min_x, sample_max_x,
                            normalized_mean_y, normalized_median_y, sample_var_y, 
                            sample_std_y, coeff_var_y, sample_min_y, sample_max_y,
                            mean_c_length_x ,median_c_length_x,min_c_length_x,max_c_length_x ,std_c_length_x,unique_categories_x,
                            mean_c_length_y ,median_c_length_y,min_c_length_y,max_c_length_y ,std_c_length_y,unique_categories_y,
                            num_identical_elements,has_shared_elements,percent_shared_elements,identical,
                            num_shared_unique_elements,has_shared_unique_elements,percent_shared_unique_elements,identical_unique,
                            correlation_value,correlation_p,ks_statistic, ks_p,
                            correlation_significant_005,ks_significant_005,
                            q_entropy_x, q_entropy_y, c_entropy_x, c_entropy_y,
                            percentage_mode_c_x, percentage_mode_c_y,
                            kurt_x, skewness_x, kurt_y, skewness_y,
                            percent_outliers_15iqr_x ,percent_outliers_3iqr_x, percent_outliers_1_99_x, percent_outliers_3std_x, 
                            has_outliers_15iqr_x, has_outliers_3iqr_x, has_outliers_1_99_x, has_outliers_3std_x,
                            percent_outliers_15iqr_y ,percent_outliers_3iqr_y, percent_outliers_1_99_y, percent_outliers_3std_y, 
                            has_outliers_15iqr_y, has_outliers_3iqr_y, has_outliers_1_99_y, has_outliers_3std_y,
                            sortedness_x, is_sorted_x, sortedness_y, is_sorted_y,
                            #plot_type
                            ]
    column_names = ['X_Type', 'Y_Type','Num_Entries_X', 'Num_Entries_Y',
            "Normalized Range (X)","Normalized Range (Y)",
            'Num_Unique_X', 'Num_Unique_Y',
            'Has_None_X', 'Num_None_X', 'Percentage_None_X',
            'Has_None_Y', 'Num_None_Y', 'Percentage_None_Y',
            "Normalized Mean (X)","Normalized Median (X)","Sample Variance (X)",
            "Sample Standard Deviation (X)","Coefficient of Variation (X)","Minimum Value (X)","Maximum Value (X)",
            "Normalized Mean (Y)","Normalized Median (Y)","Sample Variance (Y)",
            "Sample Standard Deviation (Y)","Coefficient of Variation (Y)","Minimum Value (Y)","Maximum Value (Y)",
            "Mean C (X)", "Median C (X)", "Min C (X)", "Max C (X)", "Std C (X)", "Unique C (X)",
            "Mean C (Y)", "Median C (Y)", "Min C (Y)", "Max C (Y)", "Std C (Y)", "Unique C (Y)",
            "Num Identical Elements", "Has Shared Elements", "Percent Shared Elements", "Identical", 
            "Num Shared Unique Elements", "Has Shared Unique Elements", "Percent Shared Unique Elements", "Identical Unique", 
            "Correlation Value", "Correlation P", "Ks Statistic", "Ks P",
            "Correlation Significant 005", "Ks Significant 005",
            "Q Entropy (X)", "Q Entropy (Y)", "C Entropy (X)", "C Entropy (Y)",
            "Perc Mode (X)", "Perc Mode (Y)",
            "Kurtosis (X)", "Skewness (X)", "Kurtosis (Y)", "Skewness (Y)",
            "% Outliers (1.5 IQR) (X)","% Outliers (3 IQR) (X)","% Outliers (1-99 Percentile) (X)","% Outliers (3 Std Dev) (X)",
            "Has Outliers (1.5 IQR) (X)","Has Outliers (3 IQR) (X)","Has Outliers (1-99 Percentile) (X)","Has Outliers (3 Std Dev) (X)",
            "% Outliers (1.5 IQR) (Y)","% Outliers (3 IQR) (Y)","% Outliers (1-99 Percentile) (Y)","% Outliers (3 Std Dev) (Y)",
            "Has Outliers (1.5 IQR) (Y)","Has Outliers (3 IQR) (Y)","Has Outliers (1-99 Percentile) (Y)","Has Outliers (3 Std Dev) (Y)",
            "Sortedness (X)","Is Sorted (X)","Sortedness (Y)","Is Sorted (Y)",
            #'Plot_Type'
            ]
        
    return x_values, y_values, values, column_names


def calc_features_csv(df):
    x_values = df.iloc[:, 0].values.tolist()  # First column
    y_values = df.iloc[:, 1].values.tolist() 
    
    data_type_x = categorize(x_values)
    data_type_y = categorize(y_values)
    
    #convert to numeric
    if (data_type_x == 'q'):
        try:
            x_values = convert_to_numeric_list(x_values)
        except Exception as e:
            print(e)

        if is_wrong(x_values):
            print("Found error in x values, line 57")
    if (data_type_y == 'q'):
        y_values = convert_to_numeric_list(y_values)
        leny = len(y_values)
        if is_wrong(y_values):
            print("Found error in x values, line 62")
    
    #uniqueness  features
    x_values, num_unique_x, num_entries_x, num_none_x, percentage_none_x, has_none_x = uniqueness_features(x_values)
    y_values, num_unique_y, num_entries_y, num_none_y, percentage_none_y, has_none_y = uniqueness_features(y_values)
    
    # discard data with unequal number of entries
    entry_difference = abs(num_entries_x - num_entries_y)
    max_entries = max(num_entries_x, num_entries_y)
    percentage_difference = entry_difference / max_entries
    if percentage_difference > 0.2:
        print("Error Unequal number of entries")
    if num_entries_x != num_entries_y:
        x_values , y_values = insert_missing_values(x_values, y_values, data_type_x, data_type_y)
    if (len(x_values) != len(y_values)):
        print(num_entries_x, ' ',num_entries_y)

    # general pairwise features
    num_identical_elements, has_shared_elements, percent_shared_elements, identical, num_shared_unique_elements, has_shared_unique_elements, percent_shared_unique_elements, identical_unique = g_pairwise_features(x_values,y_values)

    # statistical pairwise features
    correlation_value, correlation_p, ks_statistic, ks_p, correlation_significant_005, ks_significant_005 = s_pairwise_features(x_values,y_values,data_type_x,data_type_y)
    # has_overlap, overlap_percent

    #categorical features
    if data_type_x in ('c', 'd'):
        try:
            value_lengths_x = [len(x) for x in x_values]
            c_entropy_x, mean_c_length_x, median_c_length_x, min_c_length_x, max_c_length_x, std_c_length_x, percentage_mode_c_x, unique_categories_x = categorical_features(x_values,value_lengths_x)
        except Exception as e:
            print("Error in calculating categorical features for x")
            print("Found exception:",e)
    else:
        c_entropy_x, mean_c_length_x, median_c_length_x, min_c_length_x, max_c_length_x, std_c_length_x, percentage_mode_c_x,unique_categories_x = None, None, None, None, None, None, None, None

    if data_type_y in ('c', 'd'):
        try:
            value_lengths_y = [len(x) for x in y_values]
            c_entropy_y, mean_c_length_y, median_c_length_y, min_c_length_y, max_c_length_y, std_c_length_y, percentage_mode_c_y, unique_categories_y = categorical_features(y_values,value_lengths_y)
        except Exception as e:
            print("Error in calculating categorical features for y")
            print("Found exception:",e)
    else:
        c_entropy_y, mean_c_length_y, median_c_length_y, min_c_length_y, max_c_length_y, std_c_length_y, percentage_mode_c_y, unique_categories_y= None, None, None, None, None, None, None, None
        
    # quantitive features
    if (data_type_x == 'q'):
        sample_mean_x, sample_median_x, sample_var_x, sample_min_x, sample_max_x, sample_std_x, normalized_mean_x, normalized_median_x, coeff_var_x, range_x, normalized_range_x, q_entropy_x, kurt_x,skewness_x, percent_outliers_15iqr_x ,percent_outliers_3iqr_x, percent_outliers_1_99_x, percent_outliers_3std_x, has_outliers_15iqr_x, has_outliers_3iqr_x, has_outliers_1_99_x, has_outliers_3std_x = quantitive_features(x_values)
    else:
        sample_mean_x, sample_median_x, sample_var_x, sample_min_x, sample_max_x, sample_std_x = None, None, None, None, None, None
        normalized_mean_x, normalized_median_x, coeff_var_x, range_x, normalized_range_x, q_entropy_x = None, None, None, None, None, None
        kurt_x,skewness_x = None, None
        percent_outliers_15iqr_x ,percent_outliers_3iqr_x, percent_outliers_1_99_x, percent_outliers_3std_x, has_outliers_15iqr_x, has_outliers_3iqr_x, has_outliers_1_99_x, has_outliers_3std_x = None, None, None, None, None, None, None, None

    if (data_type_y == 'q'):
        sample_mean_y, sample_median_y, sample_var_y, sample_min_y, sample_max_y, sample_std_y, normalized_mean_y, normalized_median_y, coeff_var_y, range_y, normalized_range_y, q_entropy_y, kurt_y,skewness_y, percent_outliers_15iqr_y ,percent_outliers_3iqr_y, percent_outliers_1_99_y, percent_outliers_3std_y, has_outliers_15iqr_y, has_outliers_3iqr_y, has_outliers_1_99_y, has_outliers_3std_y = quantitive_features(y_values)
    else:
        sample_mean_y, sample_median_y, sample_var_y, sample_min_y, sample_max_y, sample_std_y = None, None, None, None, None, None
        normalized_mean_y, normalized_median_y, coeff_var_y, range_y, normalized_range_y, q_entropy_y = None, None, None, None, None, None
        kurt_y,skewness_y = None, None
        percent_outliers_15iqr_y ,percent_outliers_3iqr_y, percent_outliers_1_99_y, percent_outliers_3std_y, has_outliers_15iqr_y, has_outliers_3iqr_y, has_outliers_1_99_y, has_outliers_3std_y = None, None, None, None, None, None, None, None
    
    # sequence features
    try:
        sortedness_x, is_sorted_x = sequence_features(x_values, data_type_x)
    except Exception as e:
        print("Error found in sortedness features for x: ")
        print(e)
        sys.exit(1)
    #sortedness_y, is_sorted_y = sequence_features(y_values, data_type_y)

    try:
        sortedness_y, is_sorted_y = sequence_features(y_values, data_type_y)
    except Exception as e:
        print("Error found in sortedness features for y: ")
        print(e)
        sys.exit(1)

    values = [data_type_x, data_type_y, num_entries_x, num_entries_y,
                            normalized_range_x, normalized_range_y,
                            num_unique_x, num_unique_y,
                            has_none_x, num_none_x,percentage_none_x,
                            has_none_y, num_none_y,percentage_none_y,
                            normalized_mean_x, normalized_median_x, sample_var_x, 
                            sample_std_x, coeff_var_x, sample_min_x, sample_max_x,
                            normalized_mean_y, normalized_median_y, sample_var_y, 
                            sample_std_y, coeff_var_y, sample_min_y, sample_max_y,
                            mean_c_length_x ,median_c_length_x,min_c_length_x,max_c_length_x ,std_c_length_x,unique_categories_x,
                            mean_c_length_y ,median_c_length_y,min_c_length_y,max_c_length_y ,std_c_length_y,unique_categories_y,
                            num_identical_elements,has_shared_elements,percent_shared_elements,identical,
                            num_shared_unique_elements,has_shared_unique_elements,percent_shared_unique_elements,identical_unique,
                            correlation_value,correlation_p,ks_statistic, ks_p,
                            correlation_significant_005,ks_significant_005,
                            q_entropy_x, q_entropy_y, c_entropy_x, c_entropy_y,
                            percentage_mode_c_x, percentage_mode_c_y,
                            kurt_x, skewness_x, kurt_y, skewness_y,
                            percent_outliers_15iqr_x ,percent_outliers_3iqr_x, percent_outliers_1_99_x, percent_outliers_3std_x, 
                            has_outliers_15iqr_x, has_outliers_3iqr_x, has_outliers_1_99_x, has_outliers_3std_x,
                            percent_outliers_15iqr_y ,percent_outliers_3iqr_y, percent_outliers_1_99_y, percent_outliers_3std_y, 
                            has_outliers_15iqr_y, has_outliers_3iqr_y, has_outliers_1_99_y, has_outliers_3std_y,
                            sortedness_x, is_sorted_x, sortedness_y, is_sorted_y,
                            #plot_type
                            ]
    column_names = ['X_Type', 'Y_Type','Num_Entries_X', 'Num_Entries_Y',
            "Normalized Range (X)","Normalized Range (Y)",
            'Num_Unique_X', 'Num_Unique_Y',
            'Has_None_X', 'Num_None_X', 'Percentage_None_X',
            'Has_None_Y', 'Num_None_Y', 'Percentage_None_Y',
            "Normalized Mean (X)","Normalized Median (X)","Sample Variance (X)",
            "Sample Standard Deviation (X)","Coefficient of Variation (X)","Minimum Value (X)","Maximum Value (X)",
            "Normalized Mean (Y)","Normalized Median (Y)","Sample Variance (Y)",
            "Sample Standard Deviation (Y)","Coefficient of Variation (Y)","Minimum Value (Y)","Maximum Value (Y)",
            "Mean C (X)", "Median C (X)", "Min C (X)", "Max C (X)", "Std C (X)", "Unique C (X)",
            "Mean C (Y)", "Median C (Y)", "Min C (Y)", "Max C (Y)", "Std C (Y)", "Unique C (Y)",
            "Num Identical Elements", "Has Shared Elements", "Percent Shared Elements", "Identical", 
            "Num Shared Unique Elements", "Has Shared Unique Elements", "Percent Shared Unique Elements", "Identical Unique", 
            "Correlation Value", "Correlation P", "Ks Statistic", "Ks P",
            "Correlation Significant 005", "Ks Significant 005",
            "Q Entropy (X)", "Q Entropy (Y)", "C Entropy (X)", "C Entropy (Y)",
            "Perc Mode (X)", "Perc Mode (Y)",
            "Kurtosis (X)", "Skewness (X)", "Kurtosis (Y)", "Skewness (Y)",
            "% Outliers (1.5 IQR) (X)","% Outliers (3 IQR) (X)","% Outliers (1-99 Percentile) (X)","% Outliers (3 Std Dev) (X)",
            "Has Outliers (1.5 IQR) (X)","Has Outliers (3 IQR) (X)","Has Outliers (1-99 Percentile) (X)","Has Outliers (3 Std Dev) (X)",
            "% Outliers (1.5 IQR) (Y)","% Outliers (3 IQR) (Y)","% Outliers (1-99 Percentile) (Y)","% Outliers (3 Std Dev) (Y)",
            "Has Outliers (1.5 IQR) (Y)","Has Outliers (3 IQR) (Y)","Has Outliers (1-99 Percentile) (Y)","Has Outliers (3 Std Dev) (Y)",
            "Sortedness (X)","Is Sorted (X)","Sortedness (Y)","Is Sorted (Y)",
            #'Plot_Type'
            ]
        
    return x_values, y_values, values, column_names