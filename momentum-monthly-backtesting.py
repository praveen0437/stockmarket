import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
from datetime import timedelta

local_data_path = "nifty500_prices.pkl"  # You can also use CSV, but pickle is faster for pandas

# Load ticker list
nifty500 = pd.read_csv("nifty500.csv")
ticker_list = [f"{sym}.NS" for sym in nifty500["Symbol"].tolist()]  # Use top 50 for speed
start_date = "2020-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")


# Parameters

lookback_days = 12  # 12 months (approx trading days)
capital = 1_000_000
transaction_cost = 0.001  # 0.1% per trade (buy and sell)
rebalance_frequency = 'M'  # Monthly

if os.path.exists(local_data_path):
    data = pd.read_pickle(local_data_path)
    print("✅ Loaded existing price data.")
else:
    data = pd.DataFrame()
    print("❌ No local price data found. Starting fresh.")

# ---------- Track missing tickers ----------
existing_tickers = data.columns.tolist() if not data.empty else []
missing_tickers = list(set(ticker_list) - set(existing_tickers))

# ---------- Download missing tickers ----------
if missing_tickers:
    print(f"📥 Downloading full data for {len(missing_tickers)} missing tickers...")
    missing_data = yf.download(
        missing_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False
    )["Close"]
    data = pd.concat([data, missing_data], axis=1)
    print(f"✅ Added missing tickers: {missing_tickers}")

    
# ---------- Update existing tickers with latest data ----------
if not data.empty:
    last_date = data.index.max()
    if last_date < pd.to_datetime(end_date) - timedelta(days=1):
        fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"🔁 Updating existing tickers from {fetch_start} to {end_date}...")
        update_data = yf.download(
            existing_tickers, start=fetch_start, end=end_date, auto_adjust=True, progress=False
        )["Close"]
        data = pd.concat([data, update_data])
        data = data[~data.index.duplicated()]
    else:
        print("✅ Data is already up to date.")


# Download historical adjusted close prices
#data = yf.download(ticker_list, start=fetch_start, end=end_date, auto_adjust=True, progress=False)["Close"]
#new_data = yf.download(ticker_list, start=fetch_start, end=end_date, auto_adjust=True, progress=False)["Close"]

# ---------- Filter final tickers and clean ----------
data = data[ticker_list]  # Ensure only required tickers
data = data.sort_index()
data = data.dropna(how="all")  # Drop rows where all prices are NaN

# ---------- Save updated file ----------
data.to_pickle(local_data_path)
print(f"💾 Updated data saved to: {local_data_path}")



# Drop tickers with excessive missing data
ticker_list = [col for col in data.columns if data[col].isna().sum() < 30]
data = data[ticker_list].dropna(how="all")

# Ensure the index is datetime
data.index = pd.to_datetime(data.index)

# Resample to month-end prices
monthly_prices = data.resample('ME').last()

# Compute 12-month momentum (percent change from 12 months ago)
momentum_scores = monthly_prices.pct_change(lookback_days)

print("momentum_scores:", momentum_scores)


print("Months of data:", len(monthly_prices))

# Track portfolio history
portfolio_history = []

current_holdings = set()

# Main backtest loop: rebalance monthly after lookback period
for date in momentum_scores.index[lookback_days:]:
    scores = momentum_scores.loc[date].dropna()
    top_10 = set(scores.sort_values(ascending=False).head(10).index.tolist())
    
    if not current_holdings:
        # First month: buy all top_10
        trades = top_10
    else:
        # Buy new stocks only
        trades = top_10 - current_holdings

##    if trades:
##        print(f"🗓 {date.strftime('%Y-%m-%d')} - New Trades: {sorted(trades)}")
##    else:
##        print(f"🗓 {date.strftime('%Y-%m-%d')} - No Change in Holdings")


    # Skip if no change in holdings
    if trades == set():
        portfolio_history.append({
            "Rebalance Date": date.strftime("%Y-%m-%d"),
            "Next Date": "-",  # no trade made
            "Top Stocks": list(top_10),
            "Gross Return %": 0.0,
            "Net Return %": 0.0,
            "Portfolio Value": round(capital, 2)
        })
        continue

    # Rebalance into new top_10
    prices = monthly_prices.loc[date, list(top_10)]
    weights = capital / len(top_10) / prices
    position_values = prices * weights

    next_date_idx = monthly_prices.index.get_loc(date) + 1
    if next_date_idx >= len(monthly_prices):
        break
    next_date = monthly_prices.index[next_date_idx]
    next_prices = monthly_prices.loc[next_date, list(top_10)]

    returns = (next_prices - prices) / prices
    gross_return = (returns * weights * prices).sum() / capital

    # Calculate transaction cost only for changed positions
    n_trades = len(trades)
    transaction_cost_total = capital * transaction_cost * 2 * (n_trades / len(top_10))

    net_return = gross_return - (transaction_cost_total / capital)
    capital *= (1 + net_return)

    portfolio_history.append({
        "Rebalance Date": date.strftime("%Y-%m-%d"),
        "Next Date": next_date.strftime("%Y-%m-%d"),
        "Top Stocks": list(top_10),
        "Gross Return %": round(gross_return * 100, 2),
        "Net Return %": round(net_return * 100, 2),
        "Portfolio Value": round(capital, 2)
    })

    current_holdings = top_10
print("Top momentum scores:", scores.sort_values(ascending=False).head(10))
print("Trades:", trades)
print("current_holdings:", current_holdings)


# Convert to DataFrame
results_df = pd.DataFrame(portfolio_history)
results_df.to_csv("nse500_momentum_monthly_backtest.csv", index=False)

# Summary
total_months = len(results_df)
print("Columns in results_df:", results_df.columns)
print(results_df.tail())
avg_return = results_df["Net Return %"].mean()
cumulative_return = (capital - 1_000_000) / 1_000_000 * 100
win_rate = (results_df["Net Return %"] > 0).mean() * 100

# --- CAGR Calculation ---
start_value = 1_000_000
end_value = results_df["Portfolio Value"].iloc[-1]
years = total_months / 12
if years > 0:
    cagr = ((end_value / start_value) ** (1 / years)) - 1
else:
    cagr = np.nan

# --- Drawdown Calculation ---
portfolio_values = results_df["Portfolio Value"]
rolling_max = portfolio_values.cummax()
drawdown = (portfolio_values - rolling_max) / rolling_max
max_drawdown = drawdown.min() * 100  # as percent

print(f"\n📈 Final Monthly Momentum Strategy Summary {lookback_days} months")
print(f"Total Months: {total_months}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Average Monthly Return: {avg_return:.2f}%")
print(f"Cumulative Return: {cumulative_return:.2f}%")
print(f"CAGR: {cagr*100:.2f}%")
print(f"Max Drawdown: {max_drawdown:.2f}%")

#print("\nSample Portfolio:")
#print(results_df.head())
