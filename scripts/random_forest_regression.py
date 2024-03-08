import optuna
import pandas as pd
import numpy as np
import json

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
parameter_file = snakemake.output.parameter_file
r2_file = snakemake.output.r2_file
regression_bins = snakemake.output.regression_bins
rf_regression_balanced_input = snakemake.output.rf_regression_balanced_input

# taxon_name_col = "seq_id"
cols_to_drop = [
    "seq_id",
    "likelihood",
    "tii",
    "dataset",
    "normalised_tii",
    "rf_radius",
    "change_to_low_bootstrap_dist",
]


def train_random_forest(
    df,
    cols_to_drop,
    column_name="normalised_tii",
    cross_validate=False,
    balance_data=False,
):
    X = df.drop(cols_to_drop, axis=1)
    y = df[column_name]

    if balance_data:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=X["stability_bin"]
        )
        X_train = X_train.drop("stability_bin", axis=1)
        X_test = X_test.drop("stability_bin", axis=1)
        X = X.drop("stability_bin", axis=1)
    else:
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
    model_result = pd.DataFrame(
        {"actual": y_test}
    )
    mse = mean_squared_error(y_test, untrained_predictions)

    # print out the feature importances to file
    importances = model.feature_importances_
    pd.Series(importances, index=X.columns).to_csv(model_features_csv)

    print(f"Naive RF Mean Squared Error: {mse}")

    if cross_validate:
        # create an optimization function for optuna
        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 10, 1000)
            max_depth = trial.suggest_int("max_depth", 10, 1000, log=True)
            min_samples_split = trial.suggest_float("min_samples_split", 0.1, 1.0)
            min_samples_leaf = trial.suggest_float("min_samples_leaf", 0.1, 1.0)
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
            criterion = trial.suggest_categorical(
                "criterion",
                ["absolute_error", "poisson", "friedman_mse", "squared_error"],
            )
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                criterion=criterion,
                random_state=42,
            )
            model.fit(X_train_imputed, y_train)
            y_pred = model.predict(X_test_imputed)
            mse = mean_squared_error(y_test, y_pred)
            return mse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=200)
        best_params = study.best_params
        with open(parameter_file, "a") as f:
            json.dump(best_params, f, indent=4)
        fit_model = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            max_features=best_params["max_features"],
            criterion=best_params["criterion"],
            random_state=42,
        )
        fit_model.fit(X_train_imputed, y_train)

        fit_model_predictions = fit_model.predict(X_test_imputed)
        model_result["predicted"] = fit_model_predictions
        fit_model_importances = fit_model.feature_importances_
        pd.DataFrame(
            {
                "model_importance": fit_model_importances,
            },
            index=X.columns,
        ).to_csv(model_features_csv)
        R2 = fit_model.score(X_train_imputed, y_train)
        with open(r2_file, "w") as f:
            f.write(str(R2) + "\n")
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


def balance_df_stability_measure(df, min_test_size, bin_file, stability_measure):
    """
    Take balanced subset of df according to TII to avoid skewing predicting
    TIIs that appear most frequently in training set.
    Save number of bins and number of samples per bin in bin_file
    """
    df_copy = df.copy()  # Leave original df untouched
    reduce_binsize = True
    num_bins = 20  # Start with 20 bins, decrease if necessary
    while reduce_binsize:
        if num_bins == 1:
            reduce_binsize = False
        df_copy["stability_bin"] = pd.cut(
            df_copy[stability_measure], bins=num_bins, labels=False, duplicates="drop"
        )
        bin_counts = df_copy.groupby("stability_bin").size()
        bin_counts.to_csv(bin_file)
        min_samples_per_bin = bin_counts.min()
        if (
            min_samples_per_bin * num_bins * 0.2 < min_test_size
            or min_samples_per_bin < 2
            or num_bins
            > int(
                0.2 * min_samples_per_bin * num_bins
            )  # we need less bins than number of test samples
        ):  # we set a minimum size for our test set
            num_bins = num_bins - 1
            print(
                "Smallest bin contains ",
                min_samples_per_bin,
                " datasets. Decrease number of bins to ",
                num_bins,
            )
        else:
            reduce_binsize = False
    evenly_distributed_df = (
        df_copy.groupby("stability_bin")
        .apply(lambda x: x.sample(n=min_samples_per_bin))
        .reset_index(drop=True)
    )
    return evenly_distributed_df


balance_data = True
df = combine_dfs(csvs, subdirs)
df.to_csv(combined_csv_path)
if balance_data:
    print("Use bins to get balanced subset for regression.")
    min_test_size = 200  # needs to be adjusted to data
    df = balance_df_stability_measure(df, min_test_size, regression_bins, column_name)
    df.to_csv(rf_regression_balanced_input)
model_result = train_random_forest(
    df, cols_to_drop, column_name, cross_validate=True, balance_data=balance_data
)
model_result.to_csv(output_csv)
