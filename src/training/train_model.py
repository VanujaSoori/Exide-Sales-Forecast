import pandas as pd
import yaml
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def prepare_features(df_gold: pd.DataFrame, config: dict):
    target_col = config["forecast"]["target_column"]
    lag_days = config["forecast"]["lag_days"]

    feature_cols = ["day_of_week", "month", "is_weekend", f"lag_{lag_days}", "rolling_avg_7"]

    # Drop rows without enough history to have a valid lag/rolling value
    df_model = df_gold.dropna(subset=feature_cols)

    X = df_model[feature_cols]
    y = df_model[target_col]
    return X, y, feature_cols


def train_and_log(X, y, feature_cols, config):
    test_size = config["model"]["test_size"]

    # shuffle=False keeps chronological order — critical for time series,
    # random shuffling would leak future data into training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    with mlflow.start_run():
        params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
        }
        mlflow.log_params(params)

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mape = mean_absolute_percentage_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)

        mlflow.log_metric("mape", mape)
        mlflow.log_metric("rmse", rmse)
        mlflow.lightgbm.log_model(model, artifact_path="model")

        print(f"MAPE: {mape:.3f}  RMSE: {rmse:.3f}")

        # Baseline comparison: naive forecast = rolling_avg_7 as the prediction
        baseline_col = "rolling_avg_7"
        baseline_idx = X_test.index
        baseline_preds = X.loc[baseline_idx, baseline_col]
        baseline_mape = mean_absolute_percentage_error(y_test, baseline_preds)
        mlflow.log_metric("baseline_mape", baseline_mape)
        print(f"Baseline MAPE (rolling avg): {baseline_mape:.3f}")

        return model, mape, rmse


if __name__ == "__main__":
    config = load_config()
    df_gold = pd.read_parquet(config["paths"]["gold"])

    X, y, feature_cols = prepare_features(df_gold, config)
    model, mape, rmse = train_and_log(X, y, feature_cols, config)