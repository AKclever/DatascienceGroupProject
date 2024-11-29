import requests
import json
import pandas as pd
import os

# Define the API endpoint and the JSON request payload
url = 'https://andmed.stat.ee/api/v1/et/stat/KU091'
payload = {
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

# Send the POST request
response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))

# Check if the request was successful
if response.status_code == 200:
    # Convert the response to a Python dictionary
    data = response.json()

    # Pretty print the JSON response (optional)
    # print(json.dumps(data, indent=4))

    # Convert JSON data to a Pandas DataFrame
    # You may need to modify this line depending on the structure of the JSON data
    df = pd.json_normalize(data)

    # Define the path to save the CSV file in the 'data/raw' directory
    output_path = os.path.join("data", "raw", "theater_data2.csv")

    # Create the 'data/raw' directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
else:
    print("Failed to retrieve data. Status code:", response.status_code)
    print("Response:", response.text)
