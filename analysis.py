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
    df["Date"] = pd.DatetimeIndex(df["Date"])
    df = df.set_index("Date")
    df.drop("NaT", axis=0, inplace=True)

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Normalized", "Volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df


# Generate default EDA Report
def get_report(df):
    profile = ProfileReport(df, title=f"{df} Data Report")
    profile.to_file(f"{df}_report.html")


# Measuring volatility of entire per w/ std (dataframe, recession time period)
def std_returns(df, period, col):
    volatility = df.loc[
        ((df.index >= period[0]) & (df.index <= period[1])), f"{col}"
    ].std()

    return volatility

# rolling stats 
def rolling_stats(df, col, period=None, win_size=15):
    if period == None:
        df_col = df[col]
    else:
        df_col = df.loc[period[0] : period[1], col]

    rolling_returns = df_col.rolling(win_size)
    features = rolling_returns.aggregate(["min", "max", "mean", "std"])
    ax = features.plot()

    df_col.plot(ax=ax, color="k", alpha=0.5)
    ax.legend()
    
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

