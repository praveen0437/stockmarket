import json
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

nifty = yf.download("^NSEI", period="10d", interval="1d", auto_adjust=True, progress=False)
last_trading_day = nifty.index[-1]
print("last_trading_day",last_trading_day)
#end_date = datetime.today().strftime("%Y-%m-%d")
end_date = last_trading_day.strftime("%Y-%m-%d")
print("end_date",end_date)


# Parameters

lookback_days = 6  # 12 months (approx trading days)
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

scores = momentum_scores.loc[monthly_prices.index[-1]].dropna()

# Path to save portfolio
portfolio_file = "last_portfolio.json"

# Load last rebalance holdings
if os.path.exists(portfolio_file):
    with open(portfolio_file, "r") as f:
        last_holdings = set(json.load(f))
else:
    last_holdings = set()

# Filter top 10 positive momentum stocks
top_10 = set(scores[scores > 0].sort_values(ascending=False).head(10).index.tolist())

# Determine trades
to_buy = sorted(top_10 - last_holdings)
to_sell = sorted(last_holdings - top_10)
to_hold = sorted(last_holdings & top_10)

print("\n🧾 Rebalance Summary:")
print("✅ BUY:", to_buy)
print("❌ SELL:", to_sell)
print("🟡 HOLD:", to_hold)

# Save this rebalance for next time
with open(portfolio_file, "w") as f:
    json.dump(sorted(top_10), f)

# Optional: save to CSV for audit
pd.DataFrame({
    "Buy": pd.Series(to_buy),
    "Sell": pd.Series(to_sell),
    "Hold": pd.Series(to_hold)
}).to_csv("latest_rebalance_trades.csv", index=False)
