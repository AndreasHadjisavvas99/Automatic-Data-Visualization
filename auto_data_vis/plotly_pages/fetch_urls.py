import requests
import json
import time

def fetch_and_save_all_json(api_url, output_folder):
    page_number = 1
    #while page_number<=702:
    while page_number:   
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            # Save the current page data to a JSON file
            output_file = f'{output_folder}/page_{page_number}.json'
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Saved page {page_number} to '{output_file}'")

            # Check if there's a next page, if not, break the loop
            next_page_url = data.get('next')
            if not next_page_url:
                print("No more pages to fetch.")
                break
            
            # Update the API URL to fetch the next page
            api_url = next_page_url
            page_number += 1
        else:
            print(f"Error fetching page {page_number}: {response.status_code}")
            if response.status_code == 429:
                print("Rate limit exceeded. Waiting before retrying...")
                time.sleep(30)  # Wait for 10 seconds before retrying
            else:
                break  # Break the loop for other types of errors

# Example usage:
api_url = "https://api.plotly.com/v2/search?q=plottype%3Ascatter"  # API URL Scatter
#api_url = "https://api.plotly.com/v2/search?q=plottype%3Abar"  # API URL Bar
#api_url = "https://api.plotly.com/v2/search?q=plottype%3Apie"
#api_url = "https://api.plotly.com/v2/search?q=plottype%3Aheatmap"
#api_url = "https://api.plotly.com/v2/search?q=plottype%3Aline"
#api_url = "https://api.plotly.com/v2/search?q=plottype%3Ahistogram"
output_folder = "scatter_pages/"

fetch_and_save_all_json(api_url, output_folder)
