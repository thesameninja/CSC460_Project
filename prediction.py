import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

# Defining global economic downturn periods
dot_bomb = ["2001-03-01", "2001-11-30"]
great_rec = ["2007-12-01", "2009-06-30"]
covid = ["2020-02-01", "2020-04-30"]
trade_war = ["2018-03-22", "2018-12-31"]
trade_war_subranges = {
    "Initial Announcements": ["2018-03-22", "2018-04-02"],
    "Tariff Finalization and Start": ["2018-06-15", "2018-07-06"],
    "August-September Escalation": ["2018-08-23", "2018-09-24"],
    "December Volatility": ["2018-12-01", "2018-12-31"]
}

def to_dataframe(filename):
    """
    Load and clean the index data from a CSV file.
    The CSV file should have at least these columns:
    Date, Open, High, Low, Close, Adj Close, Normalized, Volume
    """
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename)
    df["Date"] = pd.DatetimeIndex(df["Date"])
    df = df.set_index("Date")
    # Drop rows with NaT if any
    df = df[~df.index.isnull()]

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Normalized", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)
    print(f"Data loaded successfully from {filename}!")
    return df

def get_report(df, name="Data"):
    """
    Generating an EDA report using ydata-profiling. 
    CAN CHANGE not requeired based on other group members contributions, used it as a visual
    """
    print(f"Generating EDA report for {name}...")
    profile = ProfileReport(df, title=f"{name} EDA Report", explorative=True)
    report_file = f"{name}_report.html"
    profile.to_file(report_file)
    print(f"EDA report saved to {report_file}")

def std_returns(df, period, col="Adj Close"):
    """
    Calculating the standard deviation of returns for a given period and column.
    """
    subset = df.loc[(df.index >= period[0]) & (df.index <= period[1]), col]
    return subset.pct_change().std()

def rolling_stats(df, col="Adj Close", period=None, win_size=15):
    """
    Compute and plot rolling min, max, mean, and std for a given column
    """
    if period is None:
        df_col = df[col]
    else:
        df_col = df.loc[period[0]:period[1], col]

    rolling_returns = df_col.rolling(win_size)
    features = rolling_returns.aggregate(["min", "max", "mean", "std"])
    ax = features.plot(title=f"Rolling Stats ({col}, window={win_size})", figsize=(10,6))
    df_col.plot(ax=ax, color="k", alpha=0.5)
    ax.legend()
    plt.show()

def bol_bands(df, col="Adj Close", window_size=13, num_std=2):
    """
    Compute and plot Bollinger Bands for a given column.
    """
    rolling_mean = df[col].rolling(window_size).mean()
    rolling_std = df[col].rolling(window_size).std()

    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std

    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df[col], label=f"{col} Price")
    plt.plot(df.index, rolling_mean, label="Rolling Mean", color="red")
    plt.plot(df.index, upper_band, label="Upper Band", color="black")
    plt.plot(df.index, lower_band, label="Lower Band", color="black")
    plt.fill_between(df.index, lower_band, upper_band, color="grey", alpha=0.2)
    plt.title("Bollinger Bands")
    plt.xlabel("Year")
    plt.ylabel(f"{col} Prices")
    plt.legend()
    plt.grid(True)
    plt.show()

def monte_carlo_simulation(df, period, col="Adj Close", horizon=60, num_simulations=100000):
    """
    Perform a Monte Carlo simulation to forecast potential future outcomes
    for an index assuming a downturn scenario similar to the provided period.
    Maybe in Report we explain what Monte Carlo is.

    This function:
    - Extracts historical returns during the specified downturn period.
    - Runs Monte Carlo simulations by sampling those returns with replacement.
    - Plots the distribution of final outcomes after the given horizon.
    - Prints summary statistics of final outcomes and daily returns distribution.
        Can change Number of simulations but have it set at 100k
        change the num similuations in the code to number you desire.
    """

    # Extract historical returns during the specified downturn period
    downturn_data = df.loc[(df.index >= period[0]) & (df.index <= period[1]), col]
    downturn_returns = downturn_data.pct_change().dropna()

    if downturn_returns.empty:
        print("No data available for the specified period. Cannot run Monte Carlo simulation.")
        return

    last_price = downturn_data.iloc[-1]

    # Run simulations
    simulated_final_prices = []
    all_simulated_returns = []  # Store daily returns for analysis

    for _ in range(num_simulations):
        # Randomly sample returns with replacement from the historical distribution
        sampled_returns = np.random.choice(downturn_returns, size=horizon, replace=True)
        final_price = last_price * np.prod(1 + sampled_returns)
        simulated_final_prices.append(final_price)
        all_simulated_returns.append(sampled_returns)

    simulated_final_prices = np.array(simulated_final_prices)
    all_simulated_returns = np.array(all_simulated_returns)  # shape: (num_simulations, horizon)

    # Plot the distribution of final outcomes
    plt.figure(figsize=(10,6))
    plt.hist(simulated_final_prices, bins=50, alpha=0.7, edgecolor='black')
    plt.title(f"Monte Carlo Simulation Distribution of Outcomes\n(Period: {period[0]} to {period[1]})")
    plt.xlabel("Final Simulated Price")
    plt.ylabel("Frequency")
    plt.grid(True)

    # Add summary statistics for final price
    mean_price = np.mean(simulated_final_prices)
    median_price = np.median(simulated_final_prices)
    std_dev_price = np.std(simulated_final_prices)

    plt.axvline(mean_price, color='r', linestyle='dashed', linewidth=2, label=f"Mean: {mean_price:.2f}")
    plt.axvline(median_price, color='g', linestyle='dashed', linewidth=2, label=f"Median: {median_price:.2f}")
    plt.legend()
    plt.show()

    # Print summary statistics for final outcomes
    print(f"Monte Carlo Simulation Summary ({period[0]} to {period[1]}):")
    print(f"Mean final price: {mean_price:.2f}")
    print(f"Median final price: {median_price:.2f}")
    print(f"Std Dev of final price: {std_dev_price:.2f}")

    # Daily returns statistics
    daily_mean_returns = np.mean(all_simulated_returns, axis=0)
    daily_median_returns = np.median(all_simulated_returns, axis=0)
    daily_std_returns = np.std(all_simulated_returns, axis=0)

    print("\nDaily Returns Summary Across All Simulations:")
    print(f"Mean of daily returns (averaged over {horizon} days): {np.mean(daily_mean_returns)*100:.2f}%")
    print(f"Median of daily returns (averaged over {horizon} days): {np.mean(daily_median_returns)*100:.2f}%")
    print(f"Average daily returns standard deviation: {np.mean(daily_std_returns)*100:.2f}%")

def combined_period_monte_carlo_simulation(df, periods, col="Adj Close", horizon=60, num_simulations=100000):
    """
    Perform a Monte Carlo simulation using combined downturn returns from multiple periods.
    This can simulate a more generalized downturn scenario by merging historical returns
    from all specified downturn periods. 
    Ex combingig the periods from covid and dotcom bubble bursting.

    periods: list of period lists, [dot_bomb, great_rec, covid]
    [dotcom,great reccesion and covid-19]
    """

    # Combine returns from all periods
    combined_returns = pd.Series(dtype=float)

    for period in periods:
        downturn_data = df.loc[(df.index >= period[0]) & (df.index <= period[1]), col]
        period_returns = downturn_data.pct_change().dropna()
        combined_returns = pd.concat([combined_returns, period_returns])

    if combined_returns.empty:
        print("No data available for the specified periods. Cannot run Monte Carlo simulation.")
        return

    # Use the last price from the last period as a reference
    last_period = periods[-1]
    last_price = df.loc[(df.index >= last_period[0]) & (df.index <= last_period[1]), col].iloc[-1]

    simulated_final_prices = []
    all_simulated_returns = []

    for _ in range(num_simulations):
        sampled_returns = np.random.choice(combined_returns, size=horizon, replace=True)
        final_price = last_price * np.prod(1 + sampled_returns)
        simulated_final_prices.append(final_price)
        all_simulated_returns.append(sampled_returns)

    simulated_final_prices = np.array(simulated_final_prices)
    all_simulated_returns = np.array(all_simulated_returns)  

    # Ploting final prices distribution
    plt.figure(figsize=(10,6))
    plt.hist(simulated_final_prices, bins=50, alpha=0.7, edgecolor='black')
    title_periods = " + ".join([f"{p[0]} to {p[1]}" for p in periods])
    plt.title(f"Monte Carlo Simulation Distribution of Outcomes\n(Combined Periods: {title_periods})")
    plt.xlabel("Final Simulated Price")
    plt.ylabel("Frequency")
    plt.grid(True)

    mean_price = np.mean(simulated_final_prices)
    median_price = np.median(simulated_final_prices)
    std_dev_price = np.std(simulated_final_prices)

    plt.axvline(mean_price, color='r', linestyle='dashed', linewidth=2, label=f"Mean: {mean_price:.2f}")
    plt.axvline(median_price, color='g', linestyle='dashed', linewidth=2, label=f"Median: {median_price:.2f}")
    plt.legend()
    plt.show()

    print(f"Monte Carlo Simulation Summary (Combined Periods):")
    print(f"Mean final price: {mean_price:.2f}")
    print(f"Median final price: {median_price:.2f}")
    print(f"Std Dev of final price: {std_dev_price:.2f}")

    # Daily returns statistics
    daily_mean_returns = np.mean(all_simulated_returns, axis=0)
    daily_median_returns = np.median(all_simulated_returns, axis=0)
    daily_std_returns = np.std(all_simulated_returns, axis=0)

    print("\nDaily Returns Summary Across All Simulations (Combined):")
    print(f"Mean of daily returns (averaged over {horizon} days): {np.mean(daily_mean_returns)*100:.2f}%")
    print(f"Median of daily returns (averaged over {horizon} days): {np.mean(daily_median_returns)*100:.2f}%")
    print(f"Average daily returns standard deviation: {np.mean(daily_std_returns)*100:.2f}%")

if __name__ == "__main__":
    try:
        # Load datasets for both indices
        wilshire_df = to_dataframe("wilshire_5000_data.csv")  # Ensure file is in same directory
        shanghai_df = to_dataframe("shanghai_composite_data.csv")  # Ensure file is in same directory

        # Generate EDA reports
        get_report(wilshire_df, name="Wilshire_5000")
        get_report(shanghai_df, name="Shanghai_Composite")

        # Volatility analysis during known global downturns
        downturns = {
            "Dot-Com Bubble": dot_bomb,
            "Great Recession": great_rec,
            "COVID Crash": covid
        }

        print("Volatility Analysis During Downturns:")
        for name, period in downturns.items():
            wilshire_vol = std_returns(wilshire_df, period)
            shanghai_vol = std_returns(shanghai_df, period)
            print(f"{name}:")
            print(f"  Wilshire 5000 Volatility: {wilshire_vol}")
            print(f"  Shanghai Composite Volatility: {shanghai_vol}")

        # Example rolling stats during the Great Recession for Wilshire
        print("\nRolling Stats (Wilshire 5000) During the Great Recession:")
        rolling_stats(wilshire_df, col="Adj Close", period=great_rec, win_size=20)

        # Example Bollinger Bands for Shanghai Composite
        print("Bollinger Bands (Shanghai Composite) Over Entire Period:")
        bol_bands(shanghai_df, col="Adj Close", window_size=20, num_std=2)

        # Monte Carlo Simulation for all defined scenarios (100,000 simulations)
        print("\nMonte Carlo Simulation (Wilshire 5000) - COVID-like scenario:")
        monte_carlo_simulation(wilshire_df, covid, col="Adj Close", horizon=60, num_simulations=100000)

        print("\nMonte Carlo Simulation (Shanghai Composite) - COVID-like scenario:")
        monte_carlo_simulation(shanghai_df, covid, col="Adj Close", horizon=60, num_simulations=100000)

        # Additional Monte Carlo simulations for other scenarios
        print("\nMonte Carlo Simulation (Wilshire 5000) - Dot-Com Bubble scenario:")
        monte_carlo_simulation(wilshire_df, dot_bomb, col="Adj Close", horizon=60, num_simulations=100000)

        print("\nMonte Carlo Simulation (Shanghai Composite) - Dot-Com Bubble scenario:")
        monte_carlo_simulation(shanghai_df, dot_bomb, col="Adj Close", horizon=60, num_simulations=100000)

        print("\nMonte Carlo Simulation (Wilshire 5000) - Great Recession scenario:")
        monte_carlo_simulation(wilshire_df, great_rec, col="Adj Close", horizon=60, num_simulations=100000)

        print("\nMonte Carlo Simulation (Shanghai Composite) - Great Recession scenario:")
        monte_carlo_simulation(shanghai_df, great_rec, col="Adj Close", horizon=60, num_simulations=100000)

        print("\nMonte Carlo Simulation (Wilshire 5000) - Trade War scenario:")
        monte_carlo_simulation(wilshire_df, trade_war, col="Adj Close", horizon=60, num_simulations=100000)

        print("\nMonte Carlo Simulation (Shanghai Composite) - Trade War scenario:")
        monte_carlo_simulation(shanghai_df, trade_war, col="Adj Close", horizon=60, num_simulations=100000)


        # Combined scenario (Dot-Com + Great Recession + COVID)
        print("\nMonte Carlo Simulation (Wilshire 5000) - Combined Dot-Com + Great Recession + COVID scenario:")
        combined_periods = [dot_bomb, great_rec, covid, trade_war]
        combined_period_monte_carlo_simulation(wilshire_df, combined_periods, col="Adj Close", horizon=60, num_simulations=100000)

        print("\nMonte Carlo Simulation (Shanghai Composite) - Combined Dot-Com + Great Recession + COVID scenario:")
        combined_period_monte_carlo_simulation(shanghai_df, combined_periods, col="Adj Close", horizon=60, num_simulations=100000)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
