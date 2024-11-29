import requests
import os

# URL and JSON data for the second CSV request (KU091)
url_ku091 = 'https://andmed.stat.ee/api/v1/et/stat/KU091'
json_data_ku091 = {
    "query": [
        {
            "code": "Kategooria",
            "selection": {
                "filter": "item",
                "values": ["1", "2", "3"]
            }
        },
        {
            "code": "Lavastuse liik / žanr",
            "selection": {
                "filter": "item",
                "values": [
                    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                    "11", "12", "13", "14", "15", "16", "17", "18", "19",
                    "20", "21", "22", "23", "24", "25", "26", "27", "28",
                    "29", "30", "31"
                ]
            }
        }
    ],
    "response": {
        "format": "json-stat2"
    }
}

# Path for saving the second CSV file
output_path_ku091 = 'data/raw/second_file.csv'

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_path_ku091), exist_ok=True)

# Function to download CSV data from the KU091 API
def download_csv_data(url, json_data, output_file_path):
    response = requests.post(url, json=json_data)
    if response.status_code == 200:
        with open(output_file_path, 'wb') as file:
            file.write(response.content)
        print(f"KU091 data saved to {output_file_path}")
    else:
        print(f"Failed to retrieve KU091 data. Status code: {response.status_code}")
        print(response.text)  # Print the response content for debugging if needed

# Run the function
download_csv_data(url_ku091, json_data_ku091, output_path_ku091)

