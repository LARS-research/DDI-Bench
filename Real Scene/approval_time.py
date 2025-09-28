# coding:utf-8
import requests
import pandas as pd
import time
from bs4 import BeautifulSoup

# Define request headers to simulate browser requests
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

# Proxy IP pool
proxies = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}

# Define a function to scrape drug descriptions
def get_description(db_id):
    url = f"https://go.drugbank.com/drugs/{db_id}"
    try:
        # Send GET request with headers and proxies
        response = requests.get(url, headers=headers, proxies=proxies)
        response.raise_for_status()

        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the Description tag
        description_tag = soup.find('dt', text='Description')
        if description_tag:
            description = description_tag.find_next('dd', class_='description').text.strip()
        else:
            description = 'Description Not Available'

        return description
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error fetching description for {db_id}: {e}")
        return "Error: HTTP Forbidden"
    except Exception as e:
        print(f"Error fetching description for {db_id}: {e}")
        return "Error"


def main():
    df = pd.read_excel('drug_info_with_mfmw.xlsx')

    for index, row in df.iterrows():
        # Check current row's Description
        if row['Description'] in ['Description Not Available', 'Not Found']:
            db_id = row['DB number']
            print(f"Fetching description for DB ID: {db_id}")

            description = get_description(db_id)

            # Update the Description column
            df.at[index, 'Description'] = description

            time.sleep(2)  # Add delay to prevent rate limiting

    df.to_excel('drug_info_with_approval_date.xlsx', index=False)
    print("Scraping completed")


if __name__ == "__main__":
    main()
