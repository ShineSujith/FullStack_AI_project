# FullStack_AI_project

Stock trading application made mostly using AI (prompts included).
- Uses Bollinger Band Strategy to create buy and sell signals. Buys when it goes under 1.5 standard deviation (lower band), sells when it goes above 1.5 standard (upper band), then comapres the all buy and sell singnals for a given year to plot the optimal trades.
- Creates 3 csv files one for all trades, one for optimal trades, one for a summary with number of trades awell as the average and total gain for the year.
- Only plots a graph for the optimal trades and an extra graph for equity.

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