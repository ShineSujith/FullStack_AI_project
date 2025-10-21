import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def play(stocks_dir, plots_dir, out_csv_template, year):
    """
    Simulation: Multi-stock Bollinger Band Strategy (non-overlapping optimal trade filter)
    - BUY when Adj Close <= lower Bollinger Band
    - SELL when Adj Close >= upper Bollinger Band
    - Includes:
        * Summary report per ticker (compounded gain)
        * Equity curve (cumulative return)
        * 0.1% transaction fee per trade (both sides)
        * Win/loss statistics
        * Output appended to Output.txt
    """
    theYear = year
    stocks_path = Path(stocks_dir)
    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)

    result_rows = []

    # --- Load and prepare all stock data ---
    all_data = []
    for csv_file in sorted(stocks_path.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file, parse_dates=["Date"])
        except Exception as e:
            print(f"Skipping {csv_file} - failed to read: {e}")
            continue

        if "Adj Close" not in df.columns:
            for c in df.columns:
                if c.lower().replace("_", " ") == "adj close":
                    df = df.rename(columns={c: "Adj Close"})
                    break
        if "Adj Close" not in df.columns:
            print(f"Skipping {csv_file} - no 'Adj Close' column found")
            continue

        ticker = df["Ticker"].iloc[0] if "Ticker" in df.columns else csv_file.stem.upper()

        # Compute Bollinger Bands
        df = df.sort_values("Date").reset_index(drop=True)
        df["MA20"] = df["Adj Close"].rolling(window=20, min_periods=20).mean()
        df["STD20"] = df["Adj Close"].rolling(window=20, min_periods=20).std()
        df["BB_upper"] = df["MA20"] + 1.5 * df["STD20"]
        df["BB_lower"] = df["MA20"] - 1.5 * df["STD20"]
        df["z_score"] = (df["Adj Close"] - df["MA20"]) / df["STD20"]
        df["Ticker"] = ticker

        # Keep only target year
        df_year = df[df["Date"].dt.year == theYear].copy()
        if not df_year.empty:
            all_data.append(df_year)

    if not all_data:
        print("No stock data found for the target year.")
        return

    # Combine all tickers’ data
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values("Date").reset_index(drop=True)

    # --- Simulate for each ticker independently ---
    tickers = combined["Ticker"].unique()
    fee = 0.001  # 0.1% transaction cost per side

    for ticker in tickers:
        df = combined[combined["Ticker"] == ticker].copy().reset_index(drop=True)
        invested = False
        buy_price = None
        buy_date = None

        for _, row in df.iterrows():
            # BUY
            if not invested and row["Adj Close"] <= row["BB_lower"] and not pd.isna(row["BB_lower"]):
                invested = True
                buy_date = row["Date"]
                buy_price = row["Adj Close"]
                buy_cause = f"BB_lower_cross (z={row['z_score']:.2f})"

            # SELL
            elif invested and row["Adj Close"] >= row["BB_upper"]:
                sell_date = row["Date"]
                sell_price = row["Adj Close"]
                sell_cause = "BB_upper_cross"

                # Apply transaction fees
                net_buy_price = buy_price * (1 + fee)
                net_sell_price = sell_price * (1 - fee)
                pct_gain = (net_sell_price / net_buy_price) - 1.0
                days_held = (sell_date - buy_date).days

                result_rows.append({
                    "Ticker": ticker,
                    "buy_date": buy_date.date().isoformat(),
                    "buy_price": float(buy_price),
                    "sell_date": sell_date.date().isoformat(),
                    "sell_price": float(sell_price),
                    "pct_gain": float(pct_gain),
                    "days_held": int(days_held),
                    "buy_cause": buy_cause,
                    "sell_cause": sell_cause,
                })

                invested = False
                buy_date = None
                buy_price = None

    if not result_rows:
        print("No completed trades found for the specified year.")
        return

    # --- Save all trades ---
    trades_df = pd.DataFrame(result_rows)
    trades_df["buy_date_dt"] = pd.to_datetime(trades_df["buy_date"])
    trades_df["sell_date_dt"] = pd.to_datetime(trades_df["sell_date"])
    trades_df = trades_df.sort_values(by=["buy_date_dt"])

    out_csv_all = Path(f"{theYear}_all_trades.csv")
    trades_df.to_csv(out_csv_all, index=False)
    print(f"Saved all trades -> {out_csv_all.resolve()}")

    # --- Remove overlapping trades (keep best profits only) ---
    trades_df = trades_df.sort_values(by=["pct_gain"], ascending=False)
    selected_trades = []

    for _, trade in trades_df.iterrows():
        trade_start = trade["buy_date_dt"]
        trade_end = trade["sell_date_dt"]

        overlaps = False
        for chosen in selected_trades:
            chosen_start = chosen["buy_date_dt"]
            chosen_end = chosen["sell_date_dt"]
            if not (trade_end < chosen_start or trade_start > chosen_end):
                overlaps = True
                break

        if not overlaps:
            selected_trades.append(trade)

    optimal_df = pd.DataFrame(selected_trades).sort_values("buy_date_dt")

    out_csv_opt = Path(f"{theYear}_optimal_trades.csv")
    optimal_df.to_csv(out_csv_opt, index=False)
    print(f"Saved optimal (non-overlapping) trades -> {out_csv_opt.resolve()}")

    # --- Win/Loss Statistics ---
    wins = (optimal_df["pct_gain"] > 0).sum()
    losses = (optimal_df["pct_gain"] <= 0).sum()
    total = wins + losses
    win_rate = (wins / total) * 100 if total > 0 else 0
    avg_gain = optimal_df["pct_gain"].mean()
    total_return = (1 + optimal_df["pct_gain"]).prod() - 1  # compounded

    # --- Summary Report (compounded total gain) ---
    summary_list = []
    for ticker, grp in optimal_df.groupby("Ticker"):
        num_trades = len(grp)
        avg_gain_ticker = grp["pct_gain"].mean()
        total_gain_ticker = (1 + grp["pct_gain"]).prod() - 1
        summary_list.append({
            "Ticker": ticker,
            "num_trades": num_trades,
            "avg_gain": avg_gain_ticker,
            "total_gain": total_gain_ticker
        })

    summary = pd.DataFrame(summary_list)
    total_row = {
        "Ticker": "ALL",
        "num_trades": summary["num_trades"].sum(),
        "avg_gain": summary["avg_gain"].mean(),
        "total_gain": total_return
    }
    summary = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)

    summary.to_csv(f"{theYear}_summary.csv", index=False)
    print("\n--- SUMMARY REPORT ---")
    print(summary)

    # --- Append results to Output.txt (not overwrite) ---
    output_file = Path("Output.txt")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"\n===== SUMMARY REPORT for {theYear} =====\n")
        f.write(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        f.write("\n\n--- Win/Loss Statistics ---\n")
        f.write(f"Total Trades: {total}\n")
        f.write(f"Wins: {wins}, Losses: {losses}, Win Rate: {win_rate:.2f}%\n")
        f.write(f"Average Gain per Trade: {avg_gain:.4f}\n")
        f.write(f"Total Return (with fees): {total_return:.4f}\n")
        f.write(f"Final Capital Multiple: {1 + total_return:.4f}x\n")
        f.write("=====================================\n")

    # --- Equity Curve (Cumulative Return) ---
    optimal_df = optimal_df.sort_values("sell_date_dt").copy()
    optimal_df["cum_return"] = (1 + optimal_df["pct_gain"]).cumprod() - 1

    plt.figure(figsize=(10, 5))
    plt.plot(optimal_df["sell_date_dt"], optimal_df["cum_return"], marker="o")
    plt.title(f"Cumulative Return - {theYear}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    eq_fn = plots_path / f"{theYear}_equity_curve.png"
    plt.savefig(eq_fn)
    plt.close()
    print(f"Equity curve saved -> {eq_fn}")

    # --- Plot optimal trades per ticker ---
    for ticker in optimal_df["Ticker"].unique():
        csv_candidates = list(stocks_path.glob(f"*{ticker}*.csv"))
        if not csv_candidates:
            print(f"Could not find CSV for {ticker}, skipping plot.")
            continue

        df = pd.read_csv(csv_candidates[0], parse_dates=["Date"])
        df["MA20"] = df["Adj Close"].rolling(window=20, min_periods=20).mean()
        df["STD20"] = df["Adj Close"].rolling(window=20, min_periods=20).std()
        df["BB_upper"] = df["MA20"] + 1.5 * df["STD20"]
        df["BB_lower"] = df["MA20"] - 1.5 * df["STD20"]

        plot_df = df[df["Date"].dt.year == theYear].copy()
        if plot_df.empty:
            continue

        plt.figure(figsize=(12, 6))
        plt.plot(plot_df["Date"], plot_df["Adj Close"], label="Adj Close")
        plt.plot(plot_df["Date"], plot_df["MA20"], label="MA20")
        plt.plot(plot_df["Date"], plot_df["BB_upper"], linestyle="--", label="BB_upper")
        plt.plot(plot_df["Date"], plot_df["BB_lower"], linestyle="--", label="BB_lower")

        trades = optimal_df[optimal_df["Ticker"] == ticker]
        for _, tr in trades.iterrows():
            bd = pd.to_datetime(tr["buy_date"])
            sd = pd.to_datetime(tr["sell_date"])
            plt.axvline(bd, linestyle="--", linewidth=1.2, color="green")
            plt.text(bd, plot_df["Adj Close"].max(), "BUY", rotation=90, va="bottom")
            plt.axvline(sd, linestyle="--", linewidth=1.2, color="red")
            plt.text(sd, plot_df["Adj Close"].max(), "SELL", rotation=90, va="bottom")

        plt.title(f"{ticker} - {theYear} Optimal Bollinger Band Trades")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        fn = plots_path / f"{ticker}_{theYear}_OPTIMAL.png"
        plt.savefig(fn)
        plt.close()
        print(f"Optimal plot saved for {ticker} -> {fn}")


if __name__ == "__main__":
    play()
