import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

csvs = snakemake.input.csvs
output_csv = snakemake.output.output_file_name
model_features_csv = snakemake.output.model_features_file
combined_csv_path = snakemake.output.combined_csv_path
column_name = snakemake.params.column_to_predict
subdirs = snakemake.params.subdirs

# taxon_name_col = "seq_id"
cols_to_drop = [
    "seq_id",
    "likelihood",
    "tii",
    "dataset",
    "rf_radius",
    column_name,
]


def train_random_forest(df, cols_to_drop, column_name="tii", cross_validate=False):
    X = df.drop(cols_to_drop, axis=1)
    y = df[column_name]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Train a random forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_imputed, y_train)

    # Make predictions on the test set
    untrained_predictions = model.predict(X_test_imputed)

    # Evaluate the model
    model_result = pd.DataFrame({"untuned_model_predicted": untrained_predictions, "actual": y_test})
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
        model_result["predicted"] = fit_model_predictions
        fit_model_importances = fit_model.feature_importances_
        pd.DataFrame({"untuned_model_importance": importances, "model_importance": fit_model_importances}, index=X.columns).to_csv(model_features_csv)

    return model_result


def combine_dfs(csvs, subdirs):
    df_list = []
    for subdir in subdirs:
        csv = [f for f in csvs if subdir in f][0]
        temp_df = pd.read_csv(csv, index_col=0)
        temp_df["dataset"] = subdir.split("/")[-1]
        df_list.append(temp_df)
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


df = combine_dfs(csvs, subdirs)
df.to_csv(combined_csv_path)
model_result = train_random_forest(df, cols_to_drop, column_name, cross_validate=True)
model_result.to_csv(output_csv)
