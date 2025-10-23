# FullStack_AI_project

Stock trading application made using AI GPT-5 Mini (prompt included).
The code takes in a start and end year then runs the strategy for all years between (inclusive).

## Strategy
- Buy when Adj Close is at least 1.5 STD below the 50-day MA and touches the lower Bollinger band.
- If multiple buy candidates appear same day, pick the stock furthest below its MA50.
- Sell when Adj Close is at least 1.5 STD above the 50-day MA and touches the upper Bollinger band.

## Extra notes
- Only one position at a time is held across all companies. Fees: 0.1% on buy and 0.1% on sell.
- Creates csv files for trades made during the year.
- Plots graphs for the trades and an extra graph for equity over the year.

## Setting up

- setup virtual environment on VS Code
````bash
python -m venv venv
````

Activate using:

````bash
source venv/Scripts/activate
````

- Install dependacies

````bash
pip install -r requirements.txt
````