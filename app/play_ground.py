import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set the target year
theYear = 2024

def play(stocks_dir='stocks', plots_dir='plots', out_csv_template=f'{theYear}_perf.csv'):
    """
    Modified version with ALL-OR-NOTHING investing:
    - You can only hold one stock at a time.
    - Once you buy a stock, you must wait until it sells before buying any other.
    - Trades and plots are generated normally.
    """

    stocks_path = Path(stocks_dir)
    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)

    result_rows = []

    # --- Prepare all data first ---
    all_data = []
    for csv_file in sorted(stocks_path.glob('*.csv')):
        try:
            df = pd.read_csv(csv_file, parse_dates=['Date'])
        except Exception as e:
            print(f"Skipping {csv_file} - failed to read: {e}")
            continue

        # Fix column names
        if 'Adj Close' not in df.columns:
            for c in df.columns:
                if c.lower().replace('_', ' ') == 'adj close':
                    df = df.rename(columns={c: 'Adj Close'})
                    break
        if 'Adj Close' not in df.columns:
            print(f"Skipping {csv_file} - no 'Adj Close' column found")
            continue

        # Detect ticker
        ticker = df['Ticker'].iloc[0] if 'Ticker' in df.columns else csv_file.stem.upper()

        # Compute indicators
        df = df.sort_values('Date').reset_index(drop=True)
        df['SMA100'] = df['Adj Close'].rolling(window=100, min_periods=100).mean()
        df['MA20'] = df['Adj Close'].rolling(window=20, min_periods=20).mean()
        df['STD20'] = df['Adj Close'].rolling(window=20, min_periods=20).std()
        df['BB_upper'] = df['MA20'] + 2 * df['STD20']
        df['BB_lower'] = df['MA20'] - 2 * df['STD20']
        df['pct_vs_sma100'] = df['Adj Close'] / df['SMA100'] - 1.0

        df_year = df[df['Date'].dt.year == theYear].copy()
        if df_year.empty:
            continue
        df_year['Ticker'] = ticker

        all_data.append(df_year)

    if not all_data:
        print("No stock data found for the target year.")
        return

    # --- Combine all into one big timeline for simulation ---
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values('Date').reset_index(drop=True)

    invested = False
    current_ticker = None
    buy_date = None
    buy_price = None
    pct_below = None
    buy_cause = None

    # --- Run all-or-nothing simulation ---
    for i in range(len(combined)):
        row = combined.iloc[i]
        ticker = row['Ticker']
        adj = row['Adj Close']
        sma100 = row['SMA100']
        bb_lower = row['BB_lower']
        bb_upper = row['BB_upper']
        pct_vs_sma = row['pct_vs_sma100']

        if pd.isna(sma100):
            continue

        # --- If not invested, check for BUY ---
        if not invested:
            rsi_buy = adj <= 0.70 * sma100
            bb_buy = adj <= bb_lower if not pd.isna(bb_lower) else False

            if rsi_buy or bb_buy:
                invested = True
                current_ticker = ticker
                buy_date = row['Date']
                buy_price = adj
                pct_below = -pct_vs_sma if not pd.isna(pct_vs_sma) else np.nan
                buy_cause = 'RSI' if rsi_buy else 'BB'
                # Debug print:
                # print(f"BUY {ticker} on {buy_date.date()} at {buy_price:.2f}")
        else:
            # --- If invested, only check for SELL on that same ticker ---
            if ticker != current_ticker:
                continue

            rsi_sell = adj >= 1.70 * sma100
            bb_sell = adj >= bb_upper if not pd.isna(bb_upper) else False

            if rsi_sell or bb_sell:
                sell_date = row['Date']
                sell_price = adj
                sell_cause = 'RSI' if rsi_sell else 'BB'
                pct_gain = (sell_price / buy_price) - 1.0 if buy_price else np.nan
                days_held = (sell_date - buy_date).days

                result_rows.append({
                    'Ticker': current_ticker,
                    'buy_date': buy_date.date().isoformat(),
                    'buy_price': float(buy_price),
                    'pct_below_sma100_at_buy': float(pct_below) if not pd.isna(pct_below) else np.nan,
                    'sell_date': sell_date.date().isoformat(),
                    'sell_price': float(sell_price),
                    'pct_gain': float(pct_gain),
                    'days_held': int(days_held),
                    'buy_cause': buy_cause,
                    'sell_cause': sell_cause,
                })

                # Debug print:
                # print(f"SELL {current_ticker} on {sell_date.date()} at {sell_price:.2f} | Gain: {pct_gain*100:.2f}%")

                # Reset flags
                invested = False
                current_ticker = None
                buy_date = None
                buy_price = None
                pct_below = None
                buy_cause = None

    # --- Save results ---
    if not result_rows:
        print("No completed trades found for the specified year.")
        return

    res_df = pd.DataFrame(result_rows)
    res_df['buy_date_dt'] = pd.to_datetime(res_df['buy_date'])
    res_df = res_df.sort_values('buy_date_dt').drop(columns=['buy_date_dt'])
    out_csv_path = Path(out_csv_template)
    res_df.to_csv(out_csv_path, index=False)
    print(f"Saved results to {out_csv_path.resolve()}")

    # --- Generate plots per ticker ---
    tickers = res_df['Ticker'].unique()
    for ticker in tickers:
        # Try to locate the corresponding CSV (case-insensitive match)
        csv_candidates = list(stocks_path.glob(f"*{ticker}*.csv"))
        if not csv_candidates:
            print(f"Could not find CSV for {ticker}, skipping plot.")
            continue
        csv_path = csv_candidates[0]

        try:
            df = pd.read_csv(csv_path, parse_dates=['Date'])
        except Exception as e:
            print(f"Skipping {csv_path} for plotting: {e}")
            continue

        if 'Adj Close' not in df.columns:
            for c in df.columns:
                if c.lower().replace('_', ' ') == 'adj close':
                    df = df.rename(columns={c: 'Adj Close'})
                    break

        df['SMA100'] = df['Adj Close'].rolling(window=100, min_periods=100).mean()
        plot_df = df[df['Date'].dt.year == theYear].copy()
        if plot_df.empty:
            continue

        plt.figure(figsize=(12, 6))
        plt.plot(plot_df['Date'], plot_df['Adj Close'], label='Adj Close')
        plt.plot(plot_df['Date'], plot_df['SMA100'], label='SMA100')

        trades = res_df[res_df['Ticker'] == ticker]
        for _, tr in trades.iterrows():
            bd = pd.to_datetime(tr['buy_date'])
            sd = pd.to_datetime(tr['sell_date'])
            plt.axvline(bd, linestyle='--', linewidth=1.2, color='green')
            plt.text(bd, plot_df['Adj Close'].max(), 'BUY', rotation=90, va='bottom')
            plt.axvline(sd, linestyle='--', linewidth=1.2, color='red')
            plt.text(sd, plot_df['Adj Close'].max(), 'SELL', rotation=90, va='bottom')

        plt.title(f"{ticker} - {theYear} Adj Close with SMA100 and Trades")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        fn = plots_path / f"{ticker}_{theYear}.png"
        plt.savefig(fn)
        plt.close()
        print(f"Plot saved for {ticker} -> {fn}")

# Example usage
if __name__ == "__main__":
    play()
