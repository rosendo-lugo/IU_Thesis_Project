'''
*------------------*
|                  |
|     WRANGLE!     |
|                  |
*------------------*
'''

#------------------------------------------------------------- IMPORTS  -------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import pandas_gbq
import requests
import chardet
from google.oauth2 import service_account

# print('imports loaded successfully, awaiting commands...')


#------------------------------------------------------------- ACQUIRE -------------------------------------------------------------

def check_file_exists_gbq(csv_fn, json_fn, dataset_table):
    """
    Ensures a clean dataset is available, then uploads it to BigQuery automatically.

    :param csv_fn: Filename of the local CSV (e.g., 'data.csv').
    :param json_fn: Filename of the Google Cloud service account JSON key.
    :param dataset_table: The BigQuery dataset and table (e.g., 'dataset.table_name').
    :return: A cleaned Pandas DataFrame.
    """

    # Authenticate using service account credentials
    credentials = service_account.Credentials.from_service_account_file(json_fn)

    # Step 1: Check if the CSV exists locally
    if os.path.isfile(csv_fn):
        print(f"CSV file '{csv_fn}' found locally. Loading and cleaning...")

        # Detect encoding
        with open(csv_fn, "rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            detected_encoding = result["encoding"]

        print(f"Detected encoding: {detected_encoding}")

        # Read file with detected encoding
        redfin_df = pd.read_csv(csv_fn, encoding=detected_encoding, sep="\t")

    else:
        print(f"CSV file '{csv_fn}' not found. Fetching from BigQuery...")

        # Step 2: Retrieve clean data from BigQuery
        query = f"SELECT * FROM `{dataset_table}`"
        redfin_df = pandas_gbq.read_gbq(
            query,
            project_id="iu-thesis-project",
            credentials=credentials,
            use_bqstorage_api=True
        )

        # Save locally for future use
        redfin_df.to_csv(csv_fn, index=False, encoding="utf-8")  # Save as UTF-8 for consistency
        print(f"Clean dataset fetched from BigQuery and saved locally as '{csv_fn}'")

    # Step 3: Ensure the dataset is cleaned
    redfin_df = clean_data(redfin_df)

    # Step 4: Upload cleaned data to BigQuery automatically
    upload_to_bigquery(redfin_df, dataset_table, json_fn)

    return redfin_df  # Return cleaned DataFrame
#------------------------------------------------------------- PREPARE -------------------------------------------------------------
def clean_data(df):
    """
    Cleans a Pandas DataFrame containing Redfin real estate data by standardizing column names, 
    converting data types, handling missing values, and preparing it for BigQuery upload.

    This function was originally used to clean the dataset downloaded from Redfin's website 
    (https://www.redfin.com/news/data-center/). After cleaning, the data was uploaded to BigQuery 
    for future retrieval and analysis.

    The cleaning process includes:
    - Standardizing column names (lowercase, replacing spaces/dashes with underscores).
    - Converting date columns to datetime format.
    - Handling missing values by filling them with default values.
    - Parsing currency values (e.g., "$159K" → 159000).
    - Converting percentage columns to decimal format (e.g., "5.6%" → 5.6).
    - Removing commas from numerical columns.

    :param df: A Pandas DataFrame containing Redfin real estate data.
    :return: A cleaned Pandas DataFrame, ready for analysis and BigQuery upload.
    """
    # Remove leading (and trailing) spaces from all column names
    df.columns = df.columns.str.strip()
    
    # Standardize column names: trim spaces, lowercase, replace spaces/dashes with underscores
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # Convert Month of Period End to datetime
    if "month_of_period_end" in df.columns:
        df["month_of_period_end"] = pd.to_datetime(df["month_of_period_end"], errors="coerce")

    # Define default fill values for missing data
    fill_defaults = {
        "median_sale_price": 0,
        "median_sale_price_mom": 0.0,
        "median_sale_price_yoy": 0.0,
        "homes_sold": 0,
        "homes_sold_mom": 0.0,
        "homes_sold_yoy": 0.0,
        "new_listings": 0,
        "new_listings_mom": 0.0,
        "new_listings_yoy": 0.0,
        "inventory": 0,
        "inventory_mom": 0.0,
        "inventory_yoy": 0.0,
        "days_on_market": 0,
        "average_sale_to_list": 0.0,
        "average_sale_to_list_mom": 0.0,
        "average_sale_to_list_yoy": 0.0,
        "region": "Unknown"
    }
    
    df.fillna(value=fill_defaults, inplace=True)

    # Use a dictionary to define conversions for different column types
    currency_columns = ["median_sale_price"]
    percent_columns = [
        "median_sale_price_mom", "median_sale_price_yoy", "homes_sold_mom",
        "homes_sold_yoy", "new_listings_mom", "new_listings_yoy",
        "inventory_mom", "inventory_yoy", "average_sale_to_list",
        "average_sale_to_list_mom", "average_sale_to_list_yoy"
    ]
    integer_columns = ["homes_sold", "new_listings", "inventory"]

    # Convert currency columns ($ and K notation)
    for col in currency_columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace("K", "", regex=False)
                .str.replace(",", "", regex=False)
                .replace("-", np.nan)
                .astype(int) * 1000  # Convert "159K" to 159000
            )

    # Convert percentage columns (remove % and convert to float)
    for col in percent_columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .replace("-", np.nan)
                .astype(float)
            )

    # Convert integer columns (remove commas)
    for col in integer_columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .replace("-", np.nan)
                .astype(int)
            )

    return df

#------------------------------------------------------------- UPLOAD TO BIGQUERY -------------------------------------------------------------

def upload_to_bigquery(df, dataset_table, json_fn):
    """
    Uploads a cleaned DataFrame to BigQuery.

    :param df: The cleaned Pandas DataFrame.
    :param dataset_table: The BigQuery dataset and table (iu-thesis-project.Redfin_Monthly_Housing_Market_Data.Redfin').
    :param json_fn: Filename of the Google Cloud service account JSON key.
    """

    # Authenticate using service account credentials
    credentials = service_account.Credentials.from_service_account_file(json_fn)

    # Upload to BigQuery
    pandas_gbq.to_gbq(
        df,                     # Use the function parameter instead of redfin_df
        dataset_table,          # Use the dataset_table argument
        project_id="iu-thesis-project",
        credentials=credentials,
        if_exists="replace"     # Ensures data is replaced if already exists
    )

    print(f"Cleaned data successfully uploaded to BigQuery: {dataset_table}")




# ************************************IN WORKS****************************************

import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def download_redfin_data(download_dir):
    """
    Automates the process of downloading Redfin data by selecting the format, 
    choosing Crosstab, selecting CSV, and clicking the download button.

    :param download_dir: The directory where the downloaded file should be saved.
    :return: Path to the downloaded file.
    """

    # Setup Chrome options to specify download directory
    chrome_options = webdriver.ChromeOptions()
    prefs = {"download.default_directory": download_dir}
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920x1080")
    # Remove headless mode for debugging
    # chrome_options.add_argument("--headless=new")

    # Start the WebDriver (automatically downloads the latest driver)
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Open Redfin's data page
        redfin_url = "https://www.redfin.com/news/data-center/"
        driver.get(redfin_url)

        wait = WebDriverWait(driver, 10)

        # Step 1: Click on the "Choose a format to download" button
        format_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button/span[contains(text(),'Choose a format to download')]")))
        format_button.click()
        print("Step 1: Format selection button clicked.")

        # Step 2: Select "Crosstab" from the dropdown menu
        crosstab_dropdown = wait.until(EC.element_to_be_clickable((By.XPATH, "//select[@id='data-format-selector']")))
        crosstab_dropdown.click()
        driver.execute_script("arguments[0].value = 'Crosstab'; arguments[0].dispatchEvent(new Event('change'));", crosstab_dropdown)
        print("Step 2: Crosstab format selected.")

        # Step 3: Select CSV format
        csv_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='radio' and @value='csv']")))
        driver.execute_script("arguments[0].click();", csv_button)
        print("Step 3: CSV format selected.")

        # Step 4: Click the Download button
        download_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(),'Download')]")))
        download_button.click()
        print("Step 4: Download button clicked.")

        print("Download started. Waiting for file to complete...")

        # Step 5: Wait for file to appear in the download directory
        downloaded_file = None
        for _ in range(30):  # Wait up to 30 seconds
            files = [f for f in os.listdir(download_dir) if f.endswith(".csv")]
            if files:
                downloaded_file = os.path.join(download_dir, files[0])
                break
            time.sleep(1)

        if downloaded_file:
            print(f"Download completed: {downloaded_file}")
            return downloaded_file
        else:
            raise Exception("File did not appear in the download directory within the time limit.")

    finally:
        driver.quit()  # Close the browser   

#------------------------------------------------------------- ACQUIRE 1ST VERSION -------------------------------------------------------------

def check_file_exists_gbq_V1(csv_fn, json_fn):
    """
    Ensures a clean dataset is available by:
    1. Checking for the file locally.
    2. If missing, retrieving the clean dataset from BigQuery.

    :param csv_fn: Filename of the local CSV (e.g., 'data.csv').
    :param json_fn: Filename of the Google Cloud service account JSON key.
    :return: A cleaned Pandas DataFrame.
    """

    # Authenticate using service account credentials
    credentials = service_account.Credentials.from_service_account_file(json_fn)

    # Step 1: Check if the CSV exists locally
    if os.path.isfile(csv_fn):
        print(f"CSV file '{csv_fn}' found locally. Loading and cleaning...")
        redfin_df = pd.read_csv(csv_fn, sep="\t")

    else:
        print(f"CSV file '{csv_fn}' not found. Fetching from BigQuery...")

        # Step 2: Retrieve clean data from BigQuery
        query = "SELECT * FROM `iu-thesis-project.Redfin_Monthly_Housing_Market_Data.Redfin`"
        redfin_df = pandas_gbq.read_gbq(
            query,
            project_id="iu-thesis-project",
            credentials=credentials,
            use_bqstorage_api=True
        )

        # Save locally for future use
        redfin_df.to_csv(csv_fn, index=False)
        print(f"Clean dataset fetched from BigQuery and saved locally as '{csv_fn}'")

    # Step 3: Ensure the dataset is clean before returning
    return clean_data(redfin_df)

# ************************************ IDEAS WORTH KEEPING ****************************************

# from pandas_gbq import to_gbq
# 
# to_gbq(
#     redfin_df,
#     "Redfin_Monthly_Housing_Market_Data.Redfin",  # <--- dataset.tablename
#     project_id="iu-thesis-project",               # <--- just the project name
#     if_exists="replace"
# )
