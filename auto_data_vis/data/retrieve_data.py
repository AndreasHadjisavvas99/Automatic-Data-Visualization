import os
import shutil
import json

#validate json files for correct structure and copy them to output folder

input_folders = ["../json_files/pie","../json_files/scatter","../json_files/line","../json_files/bar"]
output_folders = ["../data/pie","../data/scatter","../data/line","../data/bar"]

def has_list(lst):
    for item in lst:
        if isinstance(item, list):
            return True  # At least one element is a list
    return False
     
def correct_structure(file_path,folder_name):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            #must have 'data' key
            if 'data' not in content:
                return False
            f_data_entry = content['data'][0]
            if 'type' not in f_data_entry:
                return False
            plot_type = f_data_entry.get('type')
            if folder_name in ['scatter', 'pie', 'bar','histogram'] and folder_name != plot_type:
                return False
            if plot_type == 'pie':
                if ('labels' not in f_data_entry) or ('values' not in f_data_entry):
                    return False
                x = f_data_entry['labels']
                y = f_data_entry['values']
            elif plot_type == 'histogram':
                x = f_data_entry['x']
                y = f_data_entry['x']
            else:
                if ('x' not in f_data_entry) or ('y' not in f_data_entry):
                    return False
                
                x = f_data_entry['x']
                y = f_data_entry['y']
            # Checks if any of 'x' or 'y' is empty or the contents of 'x' or 'y' is other lists
            if not x or not y or has_list(x) or has_list(y):
                return False
            # Check if all values in 'x' or 'y' are zeros or None
            if all(value == 0 or value is None for value in x) or \
                all(value == 0 or value is None for value in y):
                return False
            # If is 'line' must have 'mode' key equal to 'lines'
            if folder_name == 'line':
                if 'mode' not in f_data_entry or f_data_entry['mode'] != "lines":
                    return False
                if plot_type == 'scatter' and f_data_entry['mode'] == "lines":
                    return True
                else:
                    return False
            return True
    except Exception as e:
        print("Error occurred:", e)
        return False
#check jsons extract features save them to data folder
def retrieve_plots(input_folder, output_folder, limit=200):
    files_copied = 0  # Counter for the number of files copied
    passed = 0
    lost = 0
    for root, dirs, files in os.walk(input_folder):
        #filename = json
        for filename in files:         
            json_file = os.path.join(root, filename)
            if not correct_structure(json_file, os.path.basename(input_folder)):
                lost += 1
                continue
            

            shutil.copy(json_file, output_folder)
            print(f"Saved {filename}")
            passed +=1
            #files_copied += 1
            #if files_copied >= limit:
            #    return
    return passed,lost

for input_folder, output_folder in zip(input_folders, output_folders):
    passed,lost = retrieve_plots(input_folder, output_folder)

print("Files skipped: ", lost)
print("Files imported: ", passed)