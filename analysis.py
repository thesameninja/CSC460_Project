import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

# Recessions DB, GR, COVID -- [Start, End]
recessions = {
    "dot_bomb": ["2001-03-01", "2001-11-30"],
    "great_rec": ["2007-12-01", "2009-06-30"],
    "trade_war" : ["2018-03-22", "2018-12-31"],
    "covid": ["2020-02-01", "2020-04-30"]
}

sns.set_theme(style="whitegrid")

def to_dataframe(filename):
    # drop first row, contains irrelevant values
    df = pd.read_csv(filename)

    # convert Date to Datetime for time series manip
    df["Date"] = pd.DatetimeIndex(df["Date"])
    df = df.set_index("Date")
    df.drop("NaT", axis=0, inplace=True)

    # Volume is constantly 0
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Normalized", "Volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df


# Generate default EDA Report
def get_report(df):
    profile = ProfileReport(df, title=f"{df["Index"][0]} Data Report")
    profile.to_file(f"{df["Index"][0]}_report.html")


# Measuring volatility of entire per w/ std (dataframe, recession time period)
def std_returns(df, period, col):
    volatility = df.loc[
        ((df.index >= period[0]) & (df.index <= period[1])), f"{col}"
    ].std()

    return volatility

# Generalized Rolling Average Function
def rolling_avg(df, col, win_size=14):
    f, axes = plt.subplots(3,1, figsize=(16,10), sharey=True)

    for ax, (period, dates) in zip(axes, recessions.items()):
        start, end = dates
        df_col = df.loc[start : end, col]

        rolling_avg = df_col.rolling(window=win_size, center=False).mean()
        ax.plot(rolling_avg, label=f'{period}')
        ax.grid(True)
        ax.set_xlabel("Date")
        ax.tick_params(labelrotation=45)
        ax.set_ylabel(f"{col}")
        ax.set_title(f"{period} Rolling Avg Over Time")
    
    f.tight_layout()

# win_size may be changed for COVID period
def rolling_std(df, col, win_size=14):
    f, axes = plt.subplots(3,1, figsize=(16,10), sharey=True)

    for ax, (period, dates) in zip(axes, recessions.items()):
        start, end = dates
        df_col = df.loc[start : end, col]

        rolling_avg = df_col.rolling(window=win_size, center=False).std()
        ax.plot(rolling_avg, label=f'{period}')
        ax.grid(True)
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{col}")
        ax.set_title(f"{period} Rolling Standard Deviation Over Time")
    
    f.tight_layout()

# distribution of Adjusted Close Prices
def close_hist(df):
    sns.histplot(df["Adj Close"], bins=13, kde=True)
    plt.title(f'{df['Index'][0]} Index Adjusted Close Prices')
    plt.xlabel("Daily Adjusted Close")
    plt.ylabel("Frequency")

# Optional
def bol_bands(df, col="Adj Close", window_size=13, num_std=2):
    rolling_mean = df[col].rolling(window_size).mean()
    rolling_std = df[col].rolling(window_size).std()

    # Calculate Bollinger Bands
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std

    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df[col], label="Index Price")
    plt.plot(
        df.index,
        rolling_mean,
        label="Rolling Mean",
        color="red",
    )
    plt.plot(df.index, upper_band, label="Upper Band", color="k")
    plt.plot(df.index, lower_band, label="Lower Band", color="k")
    plt.fill_between(
        df.index,
        lower_band,
        upper_band,
        color="grey",
        alpha=0.2,
    )
    plt.title("Bollinger Bands")
    plt.xlabel("Year")
    plt.ylabel(f"{col} Prices")
    plt.legend()
    plt.grid(True)

