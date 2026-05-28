import yfinance as yf
import pandas as pd

# SPX prices for realized vol
spx = yf.download("^GSPC", start="2010-01-01")
# VIX for implied vol
vix = yf.download("^VIX", start="2010-01-01")
