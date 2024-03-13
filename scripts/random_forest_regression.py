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
stability_measure = snakemake.params.stability_measure
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
    index, # 0: normalised_tii, 1: rf_radius
    balance_data=False,
):
    X = df.drop(cols_to_drop, axis=1)
    y = df[stability_measure[index]]

    if balance_data:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=X["stability_bin"]
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=X_train_val["stability_bin"]
            # Note: test_size=0.25 in this split will actually result in 20% of the original data being set aside for validation,
            # because 0.25 * 0.8 (remaining after first split) = 0.2
        )
        X_train = X_train.drop("stability_bin", axis=1)
        X_test = X_test.drop("stability_bin", axis=1)
        X_val = X_val.drop("stability_bin", axis=1)
    else:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42
            # Note: test_size=0.25 in this split will actually result in 20% of the original data being set aside for validation,
            # because 0.25 * 0.8 (remaining after first split) = 0.2
        )
    X = X.drop("stability_bin", axis=1) #bc we are using X's columns later

    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    # Impute missing values -- train on training set and then apply on validation and test set
    # Fit on the training data and transform it
    X_train_imputed = imputer.fit_transform(X_train)
    # Transform the validation and test sets using the same statistics
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)
    # Convert the result back to a pandas DataFrame
    X_train_imputed_df = pd.DataFrame(X_train_imputed, columns=X_train.columns, index=X_train.index)
    X_test_imputed_df = pd.DataFrame(X_test_imputed, columns=X_test.columns, index=X_test.index)

    # Hyperparameter optimisation with optuna
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 10, 1000)
        max_depth = trial.suggest_int("max_depth", 10, 1000, log=True)
        min_samples_split = trial.suggest_float("min_samples_split", 0.00001, 1.0, log=True)
        min_samples_leaf = trial.suggest_float("min_samples_leaf", 0.00001, 1.0, log=True)
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
        model.fit(X_val_imputed, y_val)
        y_pred = model.predict(X_val_imputed)
        mse = mean_squared_error(y_val, y_pred)
        return mse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200)
    best_params = study.best_params
    with open(parameter_file[index], "a") as f:
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
    model_result = pd.DataFrame(
        {
            "actual": y_test
        }
    )
    model_result["predicted"] = fit_model_predictions
    fit_model_importances = fit_model.feature_importances_
    
    pd.DataFrame(
        {
            "model_importance": fit_model_importances,
        },
        index=X.columns,
    ).to_csv(model_features_csv[index])
    R2 = fit_model.score(X_train_imputed, y_train)
    with open(r2_file[index], "w") as f:
        f.write(str(R2) + "\n")
    return model_result


def combine_dfs(csvs, subdirs):
    df_list = []
    for subdir in [subdir for subdir in subdirs]:
        csv = [f for f in csvs if subdir in f][0]
        temp_df = pd.read_csv(csv, index_col=0)
        temp_df["dataset"] = subdir.split("/")[-1]
        df_list.append(temp_df)
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


def balance_df_stability_measure(df, min_test_size, bin_file, index):
    """
    Take balanced subset of df according to TII to avoid skewing predicting
    TIIs that appear most frequently in training set.
    Save number of bins and number of samples per bin in bin_file
    index 0: normalised_tii, index 1: rf_radius
    """
    df_copy = df.copy()  # Leave original df untouched
    reduce_binsize = True
    num_bins = 20  # Start with 20 bins, decrease if necessary
    while reduce_binsize:
        if num_bins == 1:
            reduce_binsize = False
        df_copy["stability_bin"] = pd.cut(
            df_copy[stability_measure[index]], bins=num_bins, labels=False, duplicates="drop"
        )
        bin_counts = df_copy.groupby("stability_bin").size()
        bin_counts.to_csv(bin_file[index])
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


for index in [0,1]:
    balance_data = True
    df = combine_dfs(csvs, subdirs)
    df.to_csv(combined_csv_path)
    if balance_data:
        print("Use bins to get balanced subset for regression.")
        min_test_size = 200  # needs to be adjusted to data
        df = balance_df_stability_measure(df, min_test_size, regression_bins, index)
        df.to_csv(rf_regression_balanced_input[index])
    else:
        with open(regression_bins[index], "w") as f:
            pass
        with open(rf_regression_balanced_input[index], "w") as f:
            pass

    model_result = train_random_forest(
        df, cols_to_drop, index, balance_data=balance_data
    )
    model_result.to_csv(output_csv[index])