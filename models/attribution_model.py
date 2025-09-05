# models/attribution_model.py
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from ._make_features import make_features

def load_or_train(hist, carrier=None, airport=None):
    # Make features and target
    X, feature_list = make_features(hist)
    y = hist['avg_delay'].copy()

    # Drop rows with NaN in target
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Impute features with feature names preserved
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    # Train model
    model = LinearRegression()
    model.fit(X_imputed, y)

    # Save everything in a dict
    aobj = {
        'model': model,
        'imputer': imputer,
        'features': feature_list
    }
    return aobj

def predict_breakdown(aobj, row):
    # Ensure row has correct features
    X_row = pd.DataFrame([row[aobj['features']].values], columns=aobj['features'])
    X_row_imputed = pd.DataFrame(aobj['imputer'].transform(X_row), columns=aobj['features'])
    
    pred = aobj['model'].predict(X_row_imputed)[0]

    # Example breakdown logic
    total = X_row_imputed.sum(axis=1).values[0]
    breakdown = {feat: round(float(X_row_imputed[feat].values[0]/total*100), 2) 
                 for feat in aobj['features']}
    return breakdown
