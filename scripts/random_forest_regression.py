import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

column_name = snakemake.params.column_to_predict
csv = snakemake.input.csv
output_csv = snakemake.params.output_file_name
model_features_csv = snakemake.params.model_features_csv
epa_results = snakemake.input.epa_results
# taxon_name_col = "seq_id"
cols_to_drop = [
    "seq_id",
    column_name,
]


def train_random_forest(df, column_name="tii", cross_validate=False):
    X = df.drop(cols_to_drop, axis=1)
    y = df[column_name]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a random forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    untrained_predictions = model.predict(X_test)

    # Evaluate the model
    model_result = pd.DataFrame({"predicted": untrained_predictions, "actual": y_test})
    mse = mean_squared_error(y_test, untrained_predictions)

    # print out the feature importances to file
    importances = model.feature_importances_
    pd.Series(importances, index=X.columns).to_csv(model_features_csv)

    print(f"Naive RF Mean Squared Error: {mse}")

    if cross_validate:
        hyperparameter_grid={
          "n_estimators": [100,200,500,1000],
          "criterion": ["absolute_error", "poisson", "friedman_mse", "squared_error"],
          "max_depth": [10*x for x in range(1, 10)],
          "min_samples_split": [0,1,2],
          "max_features": ["sqrt", "log2", None],
          "bootstrap" : [True, False],
          "min_samples_leaf": [1, 2, 4],
          "min_samples_split": [2, 5, 10],
        }
        rf_random = RandomizedSearchCV(estimator=model,
                                       param_distributions=hyperparameter_grid,
                                       n_iter=100,
                                       cv=4,
                                       verbose=2,
                                       random_state=42,
                                       n_jobs=-1)
        rf_random.fit(X_train, y_train)
        fit_model = rf_random.best_estimator_
        fit_model_predictions = fit_model.predict(X_test)
        model_result["tuned_model_predicted"] = fit_model_predictions
        fit_model_importances = fit_model.feature_importances_
        pd.DataFrame({"importance": importances, "tuned_model_importance": fit_model_importances}, index=X.columns).to_csv(model_features_csv)

    return model_result


def create_tii_df():
    df = pd.read_csv(csv, index_col=0)
    return df


df = create_tii_df()
model_result = train_random_forest(df, column_name, cross_validate=True)
model_result.to_csv(output_csv)
