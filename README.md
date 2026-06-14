# Volatility Risk Premium

This project looks at the volatility risk premium: the gap between implied volatility from the VIX and the volatility the S&P 500 actually realizes later.

## Notebooks

- `01_data_cleaning.ipynb` loads and cleans the raw S&P 500 and VIX data.
- `02_realized_volatility.ipynb` calculates 30-day realized volatility and builds the dataset used by the app.
- `03_vrp_characterization.ipynb` explores how the volatility risk premium behaves over time.
- `04_regime_analysis.ipynb` breaks the data into market regimes like bull, bear, and crisis periods.
- `05_strategy_simulation.ipynb` runs a simple strategy simulation based on selling variance when the premium is attractive.

## Streamlit App

The Streamlit app turns the final dataset into an interactive dashboard with filters, charts, and a simple backtest view.

## Run

```bash
pip install -r requirements.txt
streamlit run main.py
```

If you need to rebuild the dataset:

```bash
python -m src.dataloader --build-master-dataset
```
