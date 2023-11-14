import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

column_name = snakemake.params.column_to_predict
csv = snakemake.input.csv
output_csv = snakemake.params.output_file_name
model_features_csv = snakemake.params.model_features_csv
epa_results = snakemake.input.epa_results
# taxon_name_col = "seq_id"
cols_to_drop = [
    "seq_id",  # we'll probably subset these columns in create_tii_df and create single values. dropped for now.
    #     "bootstrap",
    #     "order_diff",
    #     "reattachment_distances",
    #     "dist_reattachment_low_bootstrap_node",
    #     "seq_and_tree_dist_ratio",
    #     "seq_distance_ratios_closest_seq",
    column_name,
]


def train_random_forest(df, column_name):
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
    predictions = model.predict(X_test)

    # Evaluate the model
    model_result = pd.DataFrame({"predicted": predictions, "actual": y_test})
    mse = mean_squared_error(y_test, predictions)

    # print out the feature importances to file
    importances = model.feature_importances_
    pd.Series(importances, index=X.columns).to_csv(model_features_csv)

    print(f"Mean Squared Error: {mse}")
    return model_result


def create_tii_df():
    df = pd.read_csv(csv, index_col=0)
    return df


df = create_tii_df()
model_result = train_random_forest(df, column_name)
model_result.to_csv(output_csv)
