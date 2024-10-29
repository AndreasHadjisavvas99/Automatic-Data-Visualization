import os
import requests
import json
import re
from concurrent.futures import ThreadPoolExecutor

def download_json_file(output_folder, entry):
    organize_view_url = entry.get('organize_view_url')
    fid = entry.get('fid')
    fid = re.sub(r'\W+', '_', fid)
    if organize_view_url:
        download_url = f"{organize_view_url}.json"
        response = requests.get(download_url)
        if response.status_code == 200:
            download_filename = os.path.join(output_folder, f"{fid}.json")
            # Check if the file already exists
            if not os.path.exists(download_filename):
                with open(download_filename, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {download_filename}")
            else:
                print(f"File {download_filename} already exists, skipping...")
        else:
            print(f"Failed to download {download_url}: {response.status_code}")

def download_json_files(output_folder, download_folder, start_from_page=None):
    counter = 0
    start_downloading = not start_from_page
    with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers as needed
        for filename in os.listdir(output_folder):
            if filename.endswith(".json"):
                if start_from_page and filename == start_from_page:
                    start_downloading = True
                if start_downloading:
                    with open(os.path.join(output_folder, filename), 'r') as file:
                        json_data = json.load(file)
                        for entry in json_data.get('files', []):
                            executor.submit(download_json_file, download_folder, entry)
                            counter += 1

# Example usage:
output_folder = "../plotly_pages/bar_pages"  # Update with the folder where you saved the previously downloaded JSON files
download_folder = "bar"  # Update with the folder where you want to save downloaded JSON files
start_from_page = "page_1.json"
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

download_json_files(output_folder, download_folder, start_from_page)
