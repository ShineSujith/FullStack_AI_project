import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set the target year
theYear = 2024

def play(stocks_dir='stocks', plots_dir='plots', out_csv_template=f'{theYear}_perf.csv'):
    """Process all CSV files in `stocks_dir` and detect buy/sell signals for year `theYear`.

    Assumptions & interpretation (implemented):
    - CSVs have columns: Date, Close, Adj Close, High, Low, Open, Volume, Ticker
    - "RSI" in your description means the percentage difference vs the 100-day simple moving average (SMA100).
      * Buy (RSI) when: Adj Close <= 0.70 * SMA100  (i.e. at least 30% below SMA100)
      * Sell (RSI) when: Adj Close >= 1.70 * SMA100 (i.e. at least 70% above SMA100)
    - Bollinger Bands: typical 20-day moving average and +/- 2 standard deviations.
      * "Touch lower band": Adj Close <= lower_band (lower_band = ma20 - 2*std20)
      * "Touch upper band": Adj Close >= upper_band
    - Priority rule interpretation: when both RSI and Bollinger conditions occur on the same day, we mark the trigger source as "RSI" (RSI takes priority).
      However, a signal will be generated if EITHER the RSI condition OR the Bollinger band touch occurs. If both occur, RSI is noted as the cause.
    - We will look for multiple non-overlapping buy->sell trades within theYear. That is:
      * Scan chronologically through trading days in theYear.
      * When a buy signal occurs, record it and then search forward (only on later trading days) for the first sell signal; when found, record sell and close the trade.
      * Continue scanning after the sell for another buy within the same year.
    - If a buy occurs but no subsequent sell occurs within theYear, that buy is ignored (only completed trades are recorded in the output dataframe).

    Output:
    - CSV: out_csv_template (e.g. '2024_perf.csv') containing one row per completed trade with columns:
      [Ticker, buy_date, buy_price, pct_below_sma100_at_buy, sell_date, sell_price, pct_gain, days_held]
    - Plots saved to plots_dir (one PNG per ticker that had at least one completed trade). Each plot shows:
      * theYear adjusted close prices,
      * 100-day moving average,
      * buy and sell signals as vertical dashed lines (green for buy, red for sell),
      * annotated buy/sell prices on the plot.

    """

    stocks_path = Path(stocks_dir)
    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)

    result_rows = []

    # iterate over CSV files in stocks directory
    for csv_file in sorted(stocks_path.glob('*.csv')):
        try:
            df = pd.read_csv(csv_file, parse_dates=['Date'])
        except Exception as e:
            print(f"Skipping {csv_file} - failed to read: {e}")
            continue

        # Ensure expected columns
        colnames = [c.lower() for c in df.columns]
        if 'adj close' not in df.columns and 'Adj Close' not in df.columns and 'adj_close' not in df.columns:
            # try to detect possible variations
            found = [c for c in df.columns if c.lower().replace('_',' ') == 'adj close']
            if not found:
                print(f"Skipping {csv_file} - no 'Adj Close' column found")
                continue

        # Standardize column name to 'Adj Close'
        if 'Adj Close' not in df.columns:
            for c in df.columns:
                if c.lower().replace('_',' ') == 'adj close':
                    df = df.rename(columns={c: 'Adj Close'})
                    break

        # Ticker detection
        if 'Ticker' in df.columns:
            ticker = df['Ticker'].iloc[0]
        else:
            # fallback to filename without suffix
            ticker = csv_file.stem.upper()

        df = df.sort_values('Date').reset_index(drop=True)

        # compute indicators across the entire file
        df['SMA100'] = df['Adj Close'].rolling(window=100, min_periods=100).mean()
        df['MA20'] = df['Adj Close'].rolling(window=20, min_periods=20).mean()
        df['STD20'] = df['Adj Close'].rolling(window=20, min_periods=20).std()
        df['BB_upper'] = df['MA20'] + 2 * df['STD20']
        df['BB_lower'] = df['MA20'] - 2 * df['STD20']

        # percentage relative to SMA100
        df['pct_vs_sma100'] = df['Adj Close'] / df['SMA100'] - 1.0  # e.g. -0.30 means 30% below

        # Only consider trading days within theYear
        df_year = df[(df['Date'].dt.year == theYear)].copy()
        if df_year.empty:
            continue

        # we will need index alignment to original df for SMA values etc
        df_year = df_year.reset_index()

        # find signals
        buys = []  # list of dicts
        sells = []

        i = 0
        n = len(df_year)
        while i < n:
            row = df_year.loc[i]
            adj = row['Adj Close']
            sma100 = row['SMA100']
            bb_lower = row['BB_lower']
            bb_upper = row['BB_upper']
            pct_vs_sma = row['pct_vs_sma100']

            # require SMA100 to exist
            rsi_buy = False
            bb_buy = False
            if not pd.isna(sma100):
                rsi_buy = (adj <= 0.70 * sma100)
            if not pd.isna(bb_lower):
                bb_buy = (adj <= bb_lower)

            buy_signal = False
            buy_cause = None
            # Accept buy if either condition; if both, RSI takes priority
            if rsi_buy:
                buy_signal = True
                buy_cause = 'RSI'
            elif bb_buy:
                buy_signal = True
                buy_cause = 'BB'

            if buy_signal:
                buy_date = row['Date']
                buy_price = adj
                pct_below = -pct_vs_sma if (not pd.isna(pct_vs_sma)) else np.nan

                # now find the first sell after this buy (strictly later trading day)
                j = i + 1
                sell_found = False
                while j < n:
                    r2 = df_year.loc[j]
                    adj2 = r2['Adj Close']
                    sma100_2 = r2['SMA100']
                    bb_lower2 = r2['BB_lower']
                    bb_upper2 = r2['BB_upper']
                    pct_vs_sma_2 = r2['pct_vs_sma100']

                    rsi_sell = False
                    bb_sell = False
                    if not pd.isna(sma100_2):
                        rsi_sell = (adj2 >= 1.70 * sma100)
                    if not pd.isna(bb_upper2):
                        bb_sell = (adj2 >= bb_upper2)

                    sell_signal = False
                    sell_cause = None
                    if rsi_sell:
                        sell_signal = True
                        sell_cause = 'RSI'
                    elif bb_sell:
                        sell_signal = True
                        sell_cause = 'BB'

                    if sell_signal:
                        sell_date = r2['Date']
                        sell_price = adj2
                        pct_gain = (sell_price / buy_price) - 1.0 if (buy_price and not pd.isna(buy_price)) else np.nan
                        days_held = (sell_date - buy_date).days

                        result_rows.append({
                            'Ticker': ticker,
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

                        # advance i to j+1 to avoid overlapping trades
                        i = j + 1
                        sell_found = True
                        break

                    j += 1

                if not sell_found:
                    # no sell found after this buy within the year; discard this buy and continue scanning after this day
                    i += 1
            else:
                i += 1

        # if we recorded at least one trade for this ticker, create a plot for theYear
        trades_for_ticker = [r for r in result_rows if r['Ticker'] == ticker]
        if trades_for_ticker:
            # subset original df to theYear for plotting
            plot_df = df[df['Date'].dt.year == theYear].copy()
            if plot_df.empty:
                continue

            plt.figure(figsize=(12,6))
            plt.plot(plot_df['Date'], plot_df['Adj Close'], label='Adj Close')
            plt.plot(plot_df['Date'], plot_df['SMA100'], label='SMA100')

            # draw vertical dashed lines for each trade
            for tr in trades_for_ticker:
                bd = pd.to_datetime(tr['buy_date'])
                sd = pd.to_datetime(tr['sell_date'])
                plt.axvline(bd, linestyle='--', linewidth=1.2, color='green')
                plt.text(bd, plot_df['Adj Close'].max(), 'BUY', rotation=90, verticalalignment='bottom')
                plt.axvline(sd, linestyle='--', linewidth=1.2, color='red')
                plt.text(sd, plot_df['Adj Close'].max(), 'SELL', rotation=90, verticalalignment='bottom')

            plt.title(f"{ticker} - {theYear} Adj Close with SMA100 and trades")
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.tight_layout()

            fn = plots_path / f"{ticker}_{theYear}.png"
            plt.savefig(fn)
            plt.close()

    # prepare results dataframe
    if result_rows:
        res_df = pd.DataFrame(result_rows)
        # convert buy_date to datetime for sorting
        res_df['buy_date_dt'] = pd.to_datetime(res_df['buy_date'])
        res_df = res_df.sort_values('buy_date_dt').drop(columns=['buy_date_dt'])
        out_csv_path = Path(out_csv_template)
        res_df.to_csv(out_csv_path, index=False)
        print(f"Saved results to {out_csv_path.resolve()}")
    else:
        print("No completed trades found for the specified year.")