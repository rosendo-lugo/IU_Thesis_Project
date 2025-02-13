'''
*------------------*
|                  |
|     WRANGLE!     |
|                  |
*------------------*
'''

#------------------------------------------------------------- IMPORTS  -------------------------------------------------------------

import chardet  
import os
import pandas as pd
import pandas_gbq
from google.oauth2 import service_account

#------------------------------------------------------------- ACQUIRE -------------------------------------------------------------

def check_file_exists_gbq(csv_fn, json_fn, dataset_table):
    """
    Ensures a clean dataset is available.

    - If a local raw file is found, it will be cleaned and uploaded to BigQuery.
    - If the local file is missing, it will fetch an already clean version from BigQuery.
    - No unnecessary re-cleaning or re-uploading of BigQuery data.
    """

    credentials = service_account.Credentials.from_service_account_file(json_fn)

    # Step 1: Check if the CSV exists locally
    if os.path.isfile(csv_fn):
        print(f"✅ CSV file '{csv_fn}' found locally. Checking data quality...")

        # Detect encoding
        with open(csv_fn, "rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            detected_encoding = result["encoding"]

        print(f"Detected encoding: {detected_encoding}")

        # Read file with detected encoding
        redfin_df = pd.read_csv(csv_fn, encoding=detected_encoding, sep=",")

        # Step 2: Check if the data actually needs cleaning
        if needs_cleaning(redfin_df):
            print("🔹 Cleaning data before uploading...")
            redfin_df = clean_data(redfin_df)
            upload_to_bigquery(redfin_df, dataset_table, json_fn)
            print("✅ Cleaned data successfully uploaded to BigQuery.")
        else:
            print("✅ Data is already clean. No need to clean or upload.")

    else:
        print(f"⚠️ CSV file '{csv_fn}' not found. Fetching clean data from BigQuery...")

        # Step 2: Retrieve already cleaned data from BigQuery
        query = f"SELECT * FROM `{dataset_table}`"
        redfin_df = pandas_gbq.read_gbq(
            query,
            project_id="iu-thesis-project",
            credentials=credentials,
            use_bqstorage_api=True
        )

        # Step 3: Save the clean data locally for future use
        redfin_df.to_csv(csv_fn, index=False, encoding="utf-8")  # Save as UTF-8 for consistency
        print(f"✅ Clean dataset fetched from BigQuery and saved locally as '{csv_fn}'.")

    return redfin_df  # Return the dataset


#------------------------------------------------------------- ACQUIRE -------------------------------------------------------------

def check_file_exists_gbq_V2(csv_fn, json_fn, dataset_table):
    """
    Ensures a clean dataset is available. 

    - If a local raw file is found, it will be cleaned and uploaded to BigQuery.
    - If the local file is missing, it will fetch an already clean version from BigQuery.
    - No unnecessary re-cleaning or re-uploading of BigQuery data.
    """

    credentials = service_account.Credentials.from_service_account_file(json_fn)

    # Step 1: Check if the CSV exists locally
    if os.path.isfile(csv_fn):
        print(f"CSV file '{csv_fn}' found locally. Checking data quality...")

        # Detect encoding
        with open(csv_fn, "rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            detected_encoding = result["encoding"]

        print(f"Detected encoding: {detected_encoding}")

        # Read file with detected encoding
        redfin_df = pd.read_csv(csv_fn, encoding=detected_encoding, sep=",")

        # Step 2: Check if the data needs cleaning
        needs_cleaning = False
        for col in redfin_df.columns:
            if col in ["median_sale_price", "homes_sold", "new_listings", "inventory"]:
                if redfin_df[col].dtype == "object":  # Should be numeric, but it's a string
                    print(f"Column {col} should be numeric but is object. Needs cleaning!")
                    needs_cleaning = True
            elif col in ["month_of_period_end"]:
                if redfin_df[col].dtype == "object":  # Should be datetime
                    print(f"Column {col} should be datetime but is object. Needs cleaning!")
                    needs_cleaning = True

        # Step 3: If cleaning is needed, clean and upload
        if needs_cleaning:
            print("Cleaning data before uploading...")
            redfin_df = clean_data(redfin_df)
            upload_to_bigquery(redfin_df, dataset_table, json_fn)
            print("Cleaned data successfully uploaded to BigQuery.")

    else:
        print(f"CSV file '{csv_fn}' not found. Fetching clean data from BigQuery...")

        # Step 2: Retrieve already cleaned data from BigQuery
        query = f"SELECT * FROM `{dataset_table}`"
        redfin_df = pandas_gbq.read_gbq(
            query,
            project_id="iu-thesis-project",
            credentials=credentials,
            use_bqstorage_api=True
        )

        # Step 3: Save the clean data locally for future use
        redfin_df.to_csv(csv_fn, index=False, encoding="utf-8")  # Save as UTF-8 for consistency
        print(f"Clean dataset fetched from BigQuery and saved locally as '{csv_fn}'.")

    return redfin_df  # Return the dataset


#------------------------------------------------------------- PREPARE -------------------------------------------------------------
def needs_cleaning(df):
    """
    Checks if the dataset requires cleaning by validating column data types.
    Returns True if cleaning is needed, otherwise False.
    """

    # Define columns that should be numeric
    numeric_cols = [
        "median_sale_price", "homes_sold", "new_listings", "inventory",
        "median_sale_price_mom", "median_sale_price_yoy",
        "homes_sold_mom", "homes_sold_yoy",
        "new_listings_mom", "new_listings_yoy",
        "inventory_mom", "inventory_yoy",
        "average_sale_to_list", "average_sale_to_list_mom",
        "average_sale_to_list_yoy"
    ]

    # Check if numeric columns contain non-numeric values
    for col in numeric_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"⚠️ Column {col} should be numeric but is {df[col].dtype}.")
                return True  # Needs cleaning

    # Check if the date column is actually convertible to datetime
    if "month_of_period_end" in df.columns:
        try:
            df["month_of_period_end"] = pd.to_datetime(df["month_of_period_end"], errors="raise")
        except Exception:
            print(f"⚠️ Column month_of_period_end should be datetime but contains invalid values.")
            return True  # Needs cleaning

    print("✅ Data appears to be clean.")
    return False  # No cleaning needed



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
    # Standardize column names: trim spaces, lowercase, replace spaces/dashes with underscores
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(r"[^\w]", "", regex=True)  # Remove all non-alphanumeric characters
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
