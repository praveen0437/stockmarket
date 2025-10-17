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

print("momentum_scores:", momentum_scores)


print("Months of data:", len(monthly_prices))

# Track portfolio history
portfolio_history = []

current_holdings = set()
# Counters for holdings size
times_below_10 = 0
times_above_10 = 0
max_holdings = 0

# Main backtest loop: rebalance monthly after lookback period
for date in momentum_scores.index[lookback_days:]:
    scores = momentum_scores.loc[date].dropna()

    # top 10 for new buys, top 15 buffer to avoid selling winners
    top_10 = set(scores.sort_values(ascending=False).head(10).index.tolist())
    top_15 = set(scores.sort_values(ascending=False).head(15).index.tolist())

    # determine target holdings: keep current holdings that are still in top15, plus current top10
    target_holdings = (current_holdings & top_15) | top_10

    # trades to execute: buys and sells relative to current_holdings -> target_holdings
    to_buy = target_holdings - current_holdings
    to_sell = current_holdings - target_holdings
    trades = to_buy | to_sell

    # Skip if no actual trades
    if len(trades) == 0:
        portfolio_history.append({
            "Rebalance Date": date.strftime("%Y-%m-%d"),
            "Next Date": "-",  # no trade made
            "Top Stocks": list(top_10),
            "Gross Return %": 0.0,
            "Net Return %": 0.0,
            "Portfolio Value": round(capital, 2)
        })
        continue

    # Allocation and P&L calculation
    prices = monthly_prices.loc[date, list(target_holdings)]

    next_date_idx = monthly_prices.index.get_loc(date) + 1
    if next_date_idx >= len(monthly_prices):
        break
    next_date = monthly_prices.index[next_date_idx]
    next_prices = monthly_prices.loc[next_date, list(target_holdings)]

    # Determine invest ratio: keep cash if holdings < 13
    capital_before = capital
    if len(target_holdings) == 0:
        invest_ratio = 0.0
    elif len(target_holdings) < 13:
        invest_ratio = len(target_holdings) / 10.0
    else:
        invest_ratio = 1.0

    invested_capital = capital_before * invest_ratio

    if len(target_holdings) > 0 and invested_capital > 0:
        allocation_per_pos = invested_capital / len(target_holdings)
        # avoid division by zero prices
        safe_prices = prices.replace(0, np.nan)
        shares = allocation_per_pos / safe_prices
        shares = shares.fillna(0)
        pnl = (next_prices - prices) * shares
        total_pnl = pnl.sum()
    else:
        total_pnl = 0.0

    # Transaction cost approximated from invested capital
    n_trades = len(to_buy) + len(to_sell)
    if len(target_holdings) > 0:
        transaction_cost_total = invested_capital * transaction_cost * 2 * (n_trades / max(1, len(target_holdings)))
    else:
        transaction_cost_total = 0.0

    # Update capital: keep uninvested cash, add P&L, subtract transaction costs
    capital = capital_before + total_pnl - transaction_cost_total

    # Reporting returns relative to capital before rebalance
    gross_return = total_pnl / capital_before if capital_before != 0 else 0.0
    net_return = (total_pnl - transaction_cost_total) / capital_before if capital_before != 0 else 0.0

    portfolio_history.append({
        "Rebalance Date": date.strftime("%Y-%m-%d"),
        "Next Date": next_date.strftime("%Y-%m-%d"),
        "Top Stocks": list(top_10),
        "Gross Return %": round(gross_return * 100, 2),
        "Net Return %": round(net_return * 100, 2),
        "Portfolio Value": round(capital, 2),
        "To Buy": sorted(to_buy),
        "To Sell": sorted(to_sell),
        "Held (kept in buffer)": sorted(current_holdings & top_15)
    })

    # update current holdings to the target set
    current_holdings = set(target_holdings)
     # Update counters
    holding_count = len(current_holdings)
    if holding_count < 10:
        times_below_10 += 1
    if holding_count > 10:
        times_above_10 += 1
    if holding_count > max_holdings:
        max_holdings = holding_count
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
print("\nHolding size counters:")
print(f"Times holdings < 10: {times_below_10}")
print(f"Times holdings > 10: {times_above_10}")
print(f"Max holdings observed: {max_holdings}")


#print("\nSample Portfolio:")
#print(results_df.head())
