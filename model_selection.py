# model_selection.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from ._make_features import make_features


def evaluate_models(hist, test_size=0.2, random_state=42):
    """
    Train and evaluate multiple fast regression models.
    Returns a DataFrame with performance metrics.
    """

    # Create features and target
    X, feature_list = make_features(hist)
    y = hist['avg_delay'].copy()

    # Drop NaN targets
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]

    # Impute missing features
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    # Scale data for some models (like SVR, ElasticNet)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns, index=X.index)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

    # Lightweight & fast models
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=random_state),
        'Lasso': Lasso(alpha=0.001, random_state=random_state, max_iter=10000),
        'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=random_state, max_iter=10000),
        'DecisionTree': DecisionTreeRegressor(max_depth=5, random_state=random_state),
        'SVR': SVR(kernel='linear', C=1.0, epsilon=0.2)  # linear kernel = much faster
    }

    # Evaluate all models
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Model': name,
            'MAE': round(mae, 3),
            'RMSE': round(rmse, 3),
            'R2': round(r2, 3)
        })

    results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False).reset_index(drop=True)
    return results_df


def get_best_model(hist):
    """
    Returns best-performing trained model with preprocessing pipeline.
    """

    # Prepare data
    X, feature_list = make_features(hist)
    y = hist['avg_delay'].copy()
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]

    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns, index=X.index)

    # Evaluate models
    results_df = evaluate_models(hist)
    best_model_name = results_df.iloc[0]['Model']

    # Re-train the best model on all available data
    model_map = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.001, random_state=42, max_iter=10000),
        'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=10000),
        'DecisionTree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'SVR': SVR(kernel='linear', C=1.0, epsilon=0.2)
    }

    best_model = model_map[best_model_name]
    best_model.fit(X_scaled, y)

    return {
        'best_model': best_model,
        'model_name': best_model_name,
        'imputer': imputer,
        'scaler': scaler,
        'features': list(X.columns),
        'results': results_df
    }

if __name__ == "__main__":
    import pandas as pd

    # Example: load your flight dataset (update path accordingly)
    # Make sure it contains 'avg_delay' column!
    try:
        hist = pd.read_csv(r"data\flights_2025.csv")  # change path to your actual file
    except FileNotFoundError:
        print("⚠️ Please update the path to your dataset CSV in model_selection.py (line ~150).")
        exit()

    print("\n🔍 Evaluating fast regression models...\n")
    from models.model_selection import evaluate_models, get_best_model

    results = evaluate_models(hist)
    print(results.to_string(index=False))

    best = get_best_model(hist)
    print(f"\n✅ Best Model Selected: {best['model_name']}")
    print("\n📊 Detailed Results:")
    print(best['results'].to_string(index=False))