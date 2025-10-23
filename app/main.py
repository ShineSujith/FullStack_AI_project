# main.py
import os
from play_ground import play

def main():
    os.makedirs("plots", exist_ok=True)

    start_year = int(input("Enter start year (YYYY): ").strip())
    end_year = int(input("Enter end year (YYYY): ").strip())
    if end_year < start_year:
        print("End year must be >= start year.")
        return

    yearly_strategy_results = []   # (year, strategy_compounded_pct)
    market_return_rows = []        # (year, price_return_pct, total_return_pct_est)

    for year in range(start_year, end_year + 1):
        print(f"\n=== Processing year {year} ===")
        # play() returns (strategy_pct, market_price_pct, market_total_pct)
        strategy_pct, market_price_pct, market_total_pct = play(year, stock_dir="stocks", plot_dir="plots")
        yearly_strategy_results.append((year, float(strategy_pct)))
        market_return_rows.append((year, float(market_price_pct), float(market_total_pct)))
        print(f"Finished processing year {year}. Compounded gain: {strategy_pct:.2f}%")

    # Compute multi-year compounded gain for the strategy
    total_compound = 1.0
    for _, gain in yearly_strategy_results:
        total_compound *= (1.0 + gain/100.0)
    total_compounded_pct = (total_compound - 1.0) * 100.0

    # Write Results.txt (UTF-8)
    with open("Results.txt", "w", encoding="utf-8") as f:
        f.write("Strategy:\n")
        f.write("Buy when Adj Close is at least 1.5 STD below the 50-day MA and touches the lower Bollinger band.\n")
        f.write("If multiple buy candidates appear same day, pick the stock furthest below its MA50.\n")
        f.write("Sell when Adj Close is at least 1.5 STD above the 50-day MA and touches the upper Bollinger band.\n")
        f.write("Only one position at a time. Fees: 0.1% on buy and 0.1% on sell.\n\n")

        f.write("Market Price & Estimated Total Returns (calculated from an equally-weighted proxy):\n")
        f.write(f"{'Year':>6} {'Price Return (%)':>18} {'Total Return (%) (est)':>26}\n")
        for y, p_ret, t_ret in market_return_rows:
            f.write(f"{y:6d} {p_ret:18.2f}% {t_ret:26.2f}%\n")

        f.write("\nNotes on total return:\n")
        f.write("- Price Return is computed from an equally-weighted proxy built from available Adj Close prices for the year.\n")
        f.write("- Total Return (estimated) = Price Return + Estimated Dividend Yield (default 2%).\n")
        f.write("  Dividend_amount = P_start * est_div_yield; Total Return = ((P_end - P_start) + Dividend_amount) / P_start * 100.\n\n")

        f.write("Strategy yearly compounded gains (per year):\n")
        for year, gain in yearly_strategy_results:
            f.write(f"Finished processing year {year}. Compounded gain: {gain:.2f}%\n")

        f.write("\n")
        f.write(f"Total compounded gain over {end_year - start_year + 1} years: {total_compounded_pct:.2f}%\n")

    print("\nResults written to Results.txt (UTF-8).")
    print(f"Multi-year compounded gain ({start_year}-{end_year}): {total_compounded_pct:.2f}%")

if __name__ == "__main__":
    main()
