import yfinance as yf
import pandas as pd
import os

#fetchData
print("Downloading S&P 500 data...")
spx = yf.download("^GSPC", start="2010-01-01")
spx.columns = spx.columns.get_level_values(0)
spx = spx[['Close']].rename(columns={'Close': 'spx_close'})

print("Downloading VIX data...")
vix = yf.download("^VIX", start="2010-01-01")
vix.columns = vix.columns.get_level_values(0)
vix = vix[['Close']].rename(columns={'Close': 'vix_close'})

#saveData
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)
spx.to_csv(os.path.join(data_dir, "spx_raw.csv"))
vix.to_csv(os.path.join(data_dir, "vix_raw.csv"))
print("Data successfully saved to the data/ folder!")

#print
print(spx.head(3))
print(spx.tail(3))
print(vix.head(3))
print(vix.tail(3))

def verifyData(spx, vix):
    if (spx.index.equals(vix.index)):
        print("Data successfully verified!")
    else:
        print(vix.index.difference(spx.index))
        print(spx.index.difference(vix.index))

    return spx.index.equals(vix.index)


verifyData(spx, vix)


