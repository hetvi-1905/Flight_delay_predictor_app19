import pandas as pd
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

def forecast_next(hist, target_month):
    hist['month'] = pd.to_datetime(hist['month'], dayfirst=True, errors='coerce')
    hist = hist.dropna(subset=['month', 'avg_delay'])
    ts = hist.groupby('month')['avg_delay'].mean().sort_index()
    ts = ts.asfreq('MS').ffill()
    
    if len(ts) < 3:
        last_val = ts.iloc[-1] if len(ts) > 0 else 5.0
        return round(float(last_val + 1.0), 2)
    
    # Suppress SARIMAX warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(ts, order=(1,1,1), seasonal_order=(0,1,1,12))
        fobj = model.fit(disp=False)
        forecast = fobj.get_forecast(steps=1)
    
    return round(float(forecast.predicted_mean.iloc[-1]), 2)