import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

column_name = snakemake.params.column_to_predict
csvs = snakemake.input.csvs
output_csv = snakemake.output.output_file_name
model_features_csv = snakemake.output.model_features_file
combined_csv_path = snakemake.input.combined_csv_path


# taxon_name_col = "seq_id"
cols_to_drop = [
    "seq_id",
    "likelihood",
    "tii",
    "dataset",
    column_name + "_binary",
]


def train_random_forest_classifier(df, column_name="tii", cross_validate=False):
    X = df.drop(cols_to_drop, axis=1)
    y = df[column_name+"_binary"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_val = X_test[column_name]
    X_train = X_train.drop(column_name, axis=1)
    X_test = X_test.drop(column_name, axis=1)

    # Train a random forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    untrained_predictions = model.predict(X_test)

    # Evaluate the model
    model_result = pd.DataFrame({"untuned_model_predicted": untrained_predictions, "actual": y_test, "tii value": y_val})

    # print out the feature importances to file
    importances = model.feature_importances_
    pd.Series(importances, index=X_train.columns).to_csv(model_features_csv)

    if cross_validate:
        hyperparameter_grid={
          "n_estimators": [100,200,500,1000],
          "criterion": ["gini", "log-loss", None],
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
        pd.DataFrame({"untuned_model_importance": importances, "model_importance": fit_model_importances}, index=X_train.columns).to_csv(model_features_csv)

    return model_result



df = pd.read_csv(combined_csv_path)
df[column_name + "_binary"] = [True if x > 0 else False for x in df[column_name]]
df.to_csv(combined_csv_path)
model_result = train_random_forest_classifier(df, column_name, cross_validate=True)
model_result.to_csv(output_csv)
