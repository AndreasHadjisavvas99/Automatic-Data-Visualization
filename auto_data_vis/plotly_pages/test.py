import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor

def fetch_and_save_page(api_url, output_folder, page_number):
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        output_file = f'{output_folder}/page_{page_number}.json'
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved page {page_number} to '{output_file}'")
    else:
        print(f"Error fetching page {page_number}: {response.status_code}")
        if response.status_code == 429:
            print("Rate limit exceeded. Waiting before retrying...")
            time.sleep(30)  # Wait for 30 seconds before retrying

def fetch_and_save_all_json(api_url, output_folder):
    page_number = 1
    while True:
        response = requests.get(api_url)
        if response.status_code != 200:
            print(f"Error fetching page {page_number}: {response.status_code}")
            break
        
        data = response.json()
        output_file = f'{output_folder}/page_{page_number}.json'
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved page {page_number} to '{output_file}'")
        
        next_page_url = data.get('next')
        if not next_page_url:
            print("No more pages to fetch.")
            break
        
        api_url = next_page_url
        page_number += 1

def fetch_and_save_all_json_multithreaded(api_url, output_folder, num_threads=10):
    page_number = 1
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        while True:
            executor.submit(fetch_and_save_page, api_url, output_folder, page_number)
            page_number += 1
            
            response = requests.get(api_url)
            if response.status_code != 200:
                print(f"Error fetching page {page_number}: {response.status_code}")
                break
            
            data = response.json()
            next_page_url = data.get('next')
            if not next_page_url:
                print("No more pages to fetch.")
                break
            
            api_url = next_page_url

# Example usage:
api_url = "https://api.plotly.com/v2/search?q=plottype%3Abar"  # API URL Bar
output_folder = "histogram_pages_multithreaded/"

fetch_and_save_all_json_multithreaded(api_url, output_folder)
