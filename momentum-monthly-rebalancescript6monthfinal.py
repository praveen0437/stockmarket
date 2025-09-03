import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
from datetime import timedelta

# Load ticker list
nifty500 = pd.read_csv("nifty500.csv")
ticker_list = [f"{sym}.NS" for sym in nifty500["Symbol"].tolist()]  # Use top 50 for speed

nifty = yf.download("^NSEI", period="10d", interval="1d", auto_adjust=True, progress=False)
last_trading_day = nifty.index[-1]
print("last_trading_day",last_trading_day)
#start_date = "2020-01-01"
start_date = (last_trading_day - pd.DateOffset(years=1, months=6)).replace(day=1).strftime("%Y-%m-%d")
print("start_date",start_date)

#end_date = datetime.today().strftime("%Y-%m-%d")
#end_date =  "2025-07-31"
end_date = last_trading_day.strftime("%Y-%m-%d")
print("end_date",end_date)

local_data_path = f"nifty500_prices_{start_date}.pkl"  # You can also use CSV, but pickle is faster for pandas

nifty = yf.download("^NSEI", start=start_date, end=end_date, auto_adjust=True, progress=False)["Close"]
nifty_ma = nifty.rolling(200).mean()

filenameportfolio = f"portfolio-{end_date}.json"
filenameTop10 = f"top10-{end_date}.json"
filenameTop20 = f"top20-{end_date}.json"
# Parameters

lookback_months = 6  # 12 months (approx trading days)
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

# Compute lookback_months momentum (percent change from lookback_months months ago)
momentum_scores = monthly_prices.pct_change(lookback_months)

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
# Filter top 10 positive momentum stocks
top_10 = set(scores[scores > 0].sort_values(ascending=False).head(10).index.tolist())
top_20 = set(scores[scores > 0].sort_values(ascending=False).head(20).index.tolist())

# Determine trades
# Only sell if a holding drops out of the top 20
to_sell = sorted([stock for stock in last_holdings if stock not in top_20])
to_buy = sorted(top_10 - last_holdings)
to_hold = sorted((last_holdings & top_10) | (last_holdings & top_20))

not_sell = sorted([stock for stock in last_holdings if stock in top_20 and stock not in top_10])

top_after = set(scores[scores > 0].sort_values(ascending=False).head(10 - len(not_sell)).index.tolist())
not_Buy = sorted(top_10 - top_after)

print("\n🧾 Rebalance Summary:")
print("✅ BUY:", to_buy)
print("❌ SELL:", to_sell)
print("🟡 HOLD:", to_hold)
print("🟡 Not Sell these are in 20:", not_sell)
print("🟡 Not Buy if you skip to sell above:", not_Buy)

latest_portfolio = sorted(top_10 | (last_holdings & top_20))
# Save this rebalance for next time
with open(portfolio_file, "w") as f:
    json.dump(latest_portfolio, f)

# Save this rebalance for next time
# Save this rebalance for next time, including price info
portfolio_data = []
for stock in latest_portfolio:
    price = monthly_prices.loc[monthly_prices.index[-1], stock]
    momentum = scores.get(stock, float('nan'))
    portfolio_data.append({"ticker": stock, "price": price, "momentum_score": momentum})

with open(filenameportfolio, "w") as f:
    for record in portfolio_data:
        f.write(json.dumps(record) + "\n")


top10_data = []
top_10_sorted = sorted(top_10)
for stock in top_10_sorted:
    price = monthly_prices.loc[monthly_prices.index[-1], stock]
    momentum = scores.get(stock, float('nan'))
    top10_data.append({"ticker": stock, "price": price, "momentum_score": momentum})

with open(filenameTop10, "w") as f:
    for record in top10_data:
        f.write(json.dumps(record) + "\n")

top20_data = []
top_20_sorted = sorted(top_20)
for stock in top_20_sorted:
    price = monthly_prices.loc[monthly_prices.index[-1], stock]
    momentum = scores.get(stock, float('nan'))
    top20_data.append({"ticker": stock, "price": price, "momentum_score": momentum})

with open(filenameTop20, "w") as f:
    for record in top20_data:
        f.write(json.dumps(record) + "\n")


# Optional: save to CSV for audit
pd.DataFrame({
    "Buy": pd.Series(to_buy),
    "Sell": pd.Series(to_sell),
    "Hold": pd.Series(to_hold)
}).to_csv("latest_rebalance_trades.csv", index=False)


# Use asof to get the last available value on or before 'date'
nifty_value = nifty.asof(end_date)
nifty_ma_value = nifty_ma.asof(end_date)

# If result is a Series, get the first value
if isinstance(nifty_value, pd.Series):
    nifty_value = nifty_value.iloc[0]
if isinstance(nifty_ma_value, pd.Series):
    nifty_ma_value = nifty_ma_value.iloc[0]

if pd.isna(nifty_value) or pd.isna(nifty_ma_value):
    print(f"⚠️ No NIFTY data. Check 200MA manually.")    
if nifty_value < nifty_ma_value:
    print(f"⚠️ Nifty below 200 DMA Think to Trade. Market below 200-day MA.")
  
    