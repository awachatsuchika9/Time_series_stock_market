from pathlib import Path
import pandas as pd
import yfinance as yf

def download_ticker(ticker="RELIANCE.NS", start="2015-01-01", end=None, out_dir="data/raw"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.index = pd.to_datetime(df.index)
    csv_path = out_dir / f"{ticker}.csv"
    df.to_csv(csv_path)
    return csv_path, df

def prepare_close_csv(raw_csv_path, out_dir="data/processed"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_csv_path, parse_dates=True, index_col=0)
    df = df.sort_index()

    # Close price only
    df_close = df[['Close']].rename(columns={'Close':'close'})

    # ✅ Fix: Ensure numeric dtype
    df_close['close'] = pd.to_numeric(df_close['close'], errors='coerce')
    df_close = df_close.dropna()

    out_path = out_dir / (Path(raw_csv_path).stem + "_close.csv")
    df_close.to_csv(out_path)
    return out_path, df_close

def train_test_split_ts(df, test_size=0.2):
    n = len(df)
    split = int(n*(1-test_size))
    train = df.iloc[:split]
    test  = df.iloc[split:]
    return train, test
