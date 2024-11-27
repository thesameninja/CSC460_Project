import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

# Recessions DB, GR, COVID -- [Start, End]
dot_bomb = ["2001-03-01", "2001-11-30"]
great_rec = ["2007-12-01", "2009-06-30"]
covid = ["2020-02-01", "2020-04-30"]


def to_dataframe(filename):
    # drop first row, contains irrelevant values
    df = pd.read_csv(filename)

    # convert Date to Datetime for time series manip
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df = df.set_index("Date")
    df.drop("NaT", axis=0, inplace=True)

    # Expected volatility test
    df["Normalized"] = df["Normalized"].astype(float)
    df["Volume"] = df["Volume"].astype(float)

    return df


# Generate default statistics for comparisons
def get_report(df):
    profile = ProfileReport(df, title=f"{df} Data Report")
    profile.to_file(f"{df}_report.html")


# Measuring volatility w/ std (dataframe, recession time period)
def std_returns(df, period, col):
    volatility = df.loc[
        ((df.index >= period[0]) & (df.index <= period[1])), f"{col}"
    ].std()

    return volatility


# Visualization Function(s)

""" Boillinger Bands - Measures volatility based on std over 
        a period of time
         - Middle Band : Simple Moving Avg;
         - Upper/Lower : num_std above and below mean 
         - Expected win_size = 20, num_std = 1 || 2"""


def boil_bands(df, col, window_size, num_std):
    # Shift to account for NaN 1st row
    valid_indices = df.index[window_size - 1 :]

    rolling_mean = df[col].rolling(window_size).mean()
    rolling_std = df[col].rolling(window_size).std()

    # Calculate Bollinger Bands
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std

    plt.figure(figsize=(14, 7))
    plt.plot(df[col], label="Index Price")
    plt.plot(
        valid_indices,
        rolling_mean[window_size - 1 :],
        label="Rolling Mean",
        color="red",
    )
    plt.plot(
        valid_indices, upper_band[window_size - 1 :], label="Upper Band", color="blue"
    )
    plt.plot(
        valid_indices, lower_band[window_size - 1 :], label="Lower Band", color="blue"
    )
    plt.fill_between(
        valid_indices,
        lower_band[window_size - 1 :],
        upper_band[window_size - 1 :],
        color="grey",
        alpha=0.2,
    )
    plt.title(f"Bollinger Bands: {df["Index"].iloc[1]}")
    plt.xlabel("Year")
    plt.legend()
    plt.grid(True)


"""Ex.
df = to_dataframe("wilshire_5000_data.csv")
print(std_returns(df, dot_bomb, "Normalized"))
# boil_bands(df, "Normalized", 13, 2)
# plt.show()"""