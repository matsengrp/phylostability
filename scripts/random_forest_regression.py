import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

column_name = snakemake.params.column_to_predict
csv = snakemake.input.csv
epa_results = snakemake.input.epa_results
all_done_filename = snakemake.params.output_file_name
taxon_name_col="seq_id"

def train_random_forest(df, column_name):
    X = df.drop([taxon_name_col, column_name], axis=1)
    y = df[column_name]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    return mse

def create_tii_df():
    taxon_df = pd.read_csv(csv, index_col=0)
    return taxon_df

df = create_tii_df()
error = train_random_forest(df, column_name)

with open(all_done_filename, "w") as f:
    f.write(str(error))
