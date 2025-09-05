# models/_make_features.py
import pandas as pd

def make_features(df):
    """
    Returns features for attribution model
    """
    features = ['carrier_delay','weather_delay','nas_delay','security_delay','late_aircraft_delay']
    return df[features].copy(), features
