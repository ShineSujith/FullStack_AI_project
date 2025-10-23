# play_ground.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Print model version used
print("GPT Version used: GPT-5 Thinking mini")

def play(theYear, stock_dir="stocks", plot_dir="plots", ma_window=50, std_multiplier=1.5, fee_pct=0.001, est_div_yield=0.02):
    """
    Run strategy for a single year.
    Returns tuple: (strategy_compounded_pct, market_price_return_pct, market_total_return_pct_est)
    """

    os.makedirs(plot_dir, exist_ok=True)

    # Load data and compute indicators for each stock for this year
    files = [f for f in os.listdir(stock_dir) if f.endswith(".csv")]
    stock_data = {}
    for fname in files:
        path = os.path.join(stock_dir, fname)
        try:
            df_all = pd.read_csv(path, parse_dates=['Date'])
        except Exception:
            continue
        if df_all.empty:
            continue
        df_all = df_all.sort_values('Date').copy()
        dyear = df_all[df_all['Date'].dt.year == int(theYear)].copy()
        if dyear.empty or len(dyear) < ma_window:
            continue
        dyear['MA50'] = dyear['Adj Close'].rolling(ma_window).mean()
        dyear['STD50'] = dyear['Adj Close'].rolling(ma_window).std()
        dyear['Upper'] = dyear['MA50'] + std_multiplier * dyear['STD50']
        dyear['Lower'] = dyear['MA50'] - std_multiplier * dyear['STD50']
        ticker = dyear['Ticker'].iloc[0] if 'Ticker' in dyear.columns else os.path.splitext(fname)[0]
        stock_data[ticker] = dyear.reset_index(drop=True)

    if not stock_data:
        print(f"No valid stock data for year {theYear}.")
        return 0.0, 0.0, 0.0

    # Build market proxy (equally-weighted) across available tickers per date
    all_dates = sorted({d for df in stock_data.values() for d in df['Date'].tolist()})
    all_dates = pd.to_datetime(all_dates)
    proxy_rows = []
    for date in all_dates:
        vals = []
        for df in stock_data.values():
            r = df[df['Date'] == date]
            if not r.empty:
                vals.append(r['Adj Close'].values[0])
        if vals:
            proxy_rows.append((date, float(np.mean(vals))))
    proxy_df = pd.DataFrame(proxy_rows, columns=['Date', 'ProxyPrice']).sort_values('Date').reset_index(drop=True)

    if proxy_df.empty:
        market_price_return_pct = 0.0
        market_total_return_pct = 0.0
    else:
        P_start = proxy_df['ProxyPrice'].iloc[0]
        P_end = proxy_df['ProxyPrice'].iloc[-1]
        market_price_return_pct = (P_end - P_start) / P_start * 100.0
        dividend_amount = P_start * est_div_yield
        # total return interpreted as (P_end - P_start + Dividend) / P_start *100
        market_total_return_pct = ((P_end - P_start) + dividend_amount) / P_start * 100.0

    # Strategy variables
    dates = list(proxy_df['Date'])
    trades = []
    equity = 1.0
    equity_history = []  # after each sell: dict {Date, Equity}
    flag = 0
    buy_ticker = None
    buy_price = 0.0
    buy_date = None
    pct_below_ma_at_buy = 0.0

    def get_row(df, date):
        r = df[df['Date'] == date]
        return r.iloc[0] if not r.empty else None

    for current_date in dates:
        # SELL check
        if flag == 1 and buy_ticker is not None:
            df_hold = stock_data.get(buy_ticker)
            if df_hold is not None:
                row = get_row(df_hold, current_date)
                if row is not None:
                    ma = row['MA50']
                    upper = row['Upper']
                    std50 = row['STD50']
                    adj = row['Adj Close']
                    if pd.notna(ma) and pd.notna(upper) and pd.notna(std50) and buy_price > 0:
                        if (adj >= upper) and (adj >= ma + std_multiplier * std50):
                            sell_price = adj * (1.0 - fee_pct)
                            sell_date = current_date
                            pct_gain = (sell_price / buy_price - 1.0) * 100.0
                            holding_days = (sell_date - buy_date).days

                            equity *= (1.0 + pct_gain / 100.0)
                            equity_history.append({'Date': sell_date, 'Equity': equity})

                            trades.append({
                                'Ticker': buy_ticker,
                                'Buy Date': buy_date,
                                'Buy Price': round(buy_price, 6),
                                '% Below MA50': round(pct_below_ma_at_buy, 6),
                                'Sell Date': sell_date,
                                'Sell Price': round(sell_price, 6),
                                '% Gain': round(pct_gain, 6),
                                'Holding Days': holding_days,
                                'Cumulative Equity %': round((equity - 1.0) * 100.0, 6)
                            })

                            # reset
                            flag = 0
                            buy_ticker = None
                            buy_price = 0.0
                            buy_date = None
                            pct_below_ma_at_buy = 0.0
            continue

        # BUY candidates
        if flag == 0:
            candidates = []
            for ticker, df in stock_data.items():
                row = get_row(df, current_date)
                if row is None:
                    continue
                ma = row['MA50']
                lower = row['Lower']
                std50 = row['STD50']
                adj = row['Adj Close']
                if pd.isna(ma) or pd.isna(lower) or pd.isna(std50):
                    continue
                if (adj <= lower) and (adj <= ma - std_multiplier * std50):
                    pct_below = ((ma - adj) / ma) * 100.0
                    candidates.append((ticker, adj, pct_below))
            if candidates:
                best = max(candidates, key=lambda x: x[2])
                buy_ticker = best[0]
                raw_price = best[1]
                buy_price = raw_price * (1.0 + fee_pct)
                buy_date = current_date
                pct_below_ma_at_buy = best[2]
                flag = 1
                # don't record trade until sell

    # Force-close at last available date for held ticker if still holding
    if flag == 1 and buy_ticker is not None:
        df_hold = stock_data.get(buy_ticker)
        if df_hold is not None and len(df_hold) > 0:
            last_date = df_hold['Date'].max()
            row = get_row(df_hold, last_date)
            if row is not None and pd.notna(row['Adj Close']):
                adj = row['Adj Close']
                sell_price = adj * (1.0 - fee_pct)
                sell_date = last_date
                pct_gain = (sell_price / buy_price - 1.0) * 100.0
                holding_days = (sell_date - buy_date).days

                equity *= (1.0 + pct_gain / 100.0)
                equity_history.append({'Date': sell_date, 'Equity': equity})

                trades.append({
                    'Ticker': buy_ticker,
                    'Buy Date': buy_date,
                    'Buy Price': round(buy_price, 6),
                    '% Below MA50': round(pct_below_ma_at_buy, 6),
                    'Sell Date': sell_date,
                    'Sell Price': round(sell_price, 6),
                    '% Gain': round(pct_gain, 6),
                    'Holding Days': holding_days,
                    'Cumulative Equity %': round((equity - 1.0) * 100.0, 6)
                })
        # reset
        flag = 0
        buy_ticker = None
        buy_price = 0.0
        buy_date = None
        pct_below_ma_at_buy = 0.0

    # Save trades CSV and produce plots
    if trades:
        trades_df = pd.DataFrame(trades).sort_values('Buy Date').reset_index(drop=True)
        avg_return = float(trades_df['% Gain'].mean())
        wins = int((trades_df['% Gain'] > 0).sum())
        losses = int((trades_df['% Gain'] <= 0).sum())
        year_total_pct = (equity - 1.0) * 100.0

        trades_df['Avg Return % (all trades)'] = round(avg_return, 6)
        trades_df['Year Total Cumulative Return %'] = round(year_total_pct, 6)

        csv_name = f"{theYear}_perf.csv"
        trades_df.to_csv(csv_name, index=False)

        # Equity curve plot (points at sell events)
        if len(equity_history) > 0:
            eqdf = pd.DataFrame(equity_history).sort_values('Date')
            eqdf['Cumulative Return %'] = (eqdf['Equity'] - 1.0) * 100.0
            plt.figure(figsize=(10,5))
            plt.plot(eqdf['Date'], eqdf['Cumulative Return %'], marker='o')
            plt.title(f"Equity Curve (compounded) - {theYear}")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return (%)")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"equity_curve_{theYear}.png"))
            plt.close()

        # Per-stock plots (only for tickers that had trades)
        for ticker in trades_df['Ticker'].unique():
            if ticker not in stock_data:
                continue
            df_plot = stock_data[ticker]
            t_trades = trades_df[trades_df['Ticker'] == ticker]
            plt.figure(figsize=(12,6))
            plt.plot(df_plot['Date'], df_plot['Adj Close'], label='Adj Close')
            plt.plot(df_plot['Date'], df_plot['MA50'], linestyle='--', label='MA50')
            plt.plot(df_plot['Date'], df_plot['Upper'], linestyle=':', label='Upper')
            plt.plot(df_plot['Date'], df_plot['Lower'], linestyle=':', label='Lower')
            for _, r in t_trades.iterrows():
                plt.axvline(r['Buy Date'], color='red', linestyle='--', alpha=0.7)
                plt.axvline(r['Sell Date'], color='green', linestyle='--', alpha=0.7)
                try:
                    plt.scatter([r['Buy Date']], [r['Buy Price']], color='red', marker='^', zorder=5)
                    plt.scatter([r['Sell Date']], [r['Sell Price']], color='green', marker='v', zorder=5)
                except Exception:
                    pass
            plt.title(f"{ticker} - {theYear}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{ticker}_{theYear}.png"))
            plt.close()

        # Print summary
        print(f"Year {theYear} summary:")
        print(f"Number of trades: {len(trades_df)}")
        print(f"Winning trades: {wins}")
        print(f"Losing trades: {losses}")
        print(f"Average return per trade: {round(avg_return,4)}%")
        print(f"Year total compounded return: {round(year_total_pct,4)}%")

        return round(year_total_pct, 6), round(market_price_return_pct, 6), round(market_total_return_pct, 6)
    else:
        # No trades executed
        print(f"No trades executed for {theYear}.")
        # still return market figures
        return 0.0, round(market_price_return_pct, 6), round(market_total_return_pct, 6)
