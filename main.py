import yfinance as yf
import pandas as pd

def fetch_data(ticker, start_date, end_date):
    """
    Fetch historical data for a given ticker from Yahoo Finance.
    
    Parameters:
    ticker (str): Ticker symbol for the index.
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
    pd.DataFrame: Historical data with reset index, or None if the data fetch fails.
    """
    print(f"Fetching data for {ticker}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}.")
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def preprocess_data(data, name):
    """
    Preprocess the data by adding normalized prices and daily returns.
    
    Parameters:
    data (pd.DataFrame): Dataframe with raw historical data.
    name (str): Name of the index for tagging.
    
    Returns:
    pd.DataFrame: Preprocessed dataframe.
    """
    if data is None or data.empty:
        print(f"No data to preprocess for {name}. Skipping...")
        return None
    
    print(f"Preprocessing data for {name}...")
    # Fill missing values
    data.ffill(inplace=True)
    data.bfill(inplace=True)
    
    # Add normalized prices
    data["Normalized"] = data["Adj Close"] / data["Adj Close"].iloc[0]
    
    # Add daily returns
    data["Daily Return"] = data["Adj Close"].pct_change()
    
    # Add index metadata
    data["Index"] = name
    
    return data

def save_to_csv(data, filename):
    """
    Save the dataframe to a CSV file.
    
    Parameters:
    data (pd.DataFrame): Dataframe to save.
    filename (str): File name for the CSV.
    """
    if data is None or data.empty:
        print(f"No data to save for {filename}. Skipping...")
        return
    
    print(f"Saving data to {filename}...")
    data.to_csv(filename, index=False)
    print(f"Data saved successfully to {filename}.")

def main():
    # Define the parameters
    indices = {
        "Wilshire 5000": "^W5000",
        "Shanghai Composite": "000001.SS"
    }
    start_date = "2000-01-01"
    end_date = "2024-01-01"
    
    # Fetch, preprocess, and save data for each index
    for name, ticker in indices.items():
        # Fetch data
        data = fetch_data(ticker, start_date, end_date)
        
        # Preprocess data
        processed_data = preprocess_data(data, name)
        
        # Save to CSV
        filename = f"{name.replace(' ', '_').lower()}_data.csv"
        save_to_csv(processed_data, filename)
    
    print("All data processing completed!")

if __name__ == "__main__":
    main()
