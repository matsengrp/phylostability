import optuna
import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, accuracy_score

combined_statistics = snakemake.input.combined_statistics
au_test_results = snakemake.input.au_test_results
output_csv = snakemake.params.output_file_name
model_features_csv = snakemake.output.model_features_file
classifier_metrics_csv = snakemake.output.classifier_metrics_csv
parameter_file = snakemake.params.parameter_file
plot_folder = snakemake.params.plot_folder
combined_csv_path = snakemake.params.combined_csv_path


# taxon_name_col = "seq_id"
cols_to_drop = [
    "seq_id",
    "likelihood",
    "normalised_tii",
    "dataset",
    "rf_radius",
    "tii"
    # "p-AU_binary",
]


def train_random_forest_classifier(df, column_name, cross_validate=False):
    X = df.drop(cols_to_drop, axis=1)
    y = df[column_name]
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_val = X_test[column_name]
    X_train = X_train.drop(column_name, axis=1)
    X_test = X_test.drop(column_name, axis=1)

    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Train a random forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_imputed, y_train)

    # Make predictions on the test set
    untrained_predictions = model.predict(X_test_imputed)

    # Evaluate the model
    model_result = pd.DataFrame(
        {
            "untuned_model_predicted": untrained_predictions,
            "actual": y_test,
            "predicted value": y_val,
        }
    )

    importances = model.feature_importances_
    pd.Series(importances, index=X_train.columns).to_csv(model_features_csv)

    y_scores = model.predict_proba(X_test_imputed)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(classifier_metrics_csv)
    if cross_validate:
        # create an optimization function for optuna
        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 10, 1000)
            max_depth = trial.suggest_int("max_depth", 10, 1000, log=True)
            min_samples_split = trial.suggest_float("min_samples_split", 0.1, 1.0)
            min_samples_leaf = trial.suggest_float("min_samples_leaf", 0.1, 1.0)
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
            criterion = trial.suggest_categorical(
                "criterion", ["gini", "log_loss", "entropy"]
            )
            model = RandomForestClassifier(
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
            accuracy = accuracy_score(y_test, y_pred)
            return accuracy

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=200)
        best_params = study.best_params
        with open(parameter_file, "a") as f:
            json.dump(best_params, f, indent=4)
        fit_model = RandomForestClassifier(
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
        y_scores = fit_model.predict_proba(X_test_imputed)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(classifier_metrics_csv)
        pd.DataFrame(
            {
                "untuned_model_importance": importances,
                "model_importance": fit_model_importances,
            },
            index=X_train.columns,
        ).to_csv(model_features_csv)

    return model_result


def add_au_test_result(df, au_df, only_au = False):
    """
    Add results from au_test in given file au_test_results to df conrtaining all summary statistics
    """
    au_df_subset = au_df[["seq_id", "dataset", "p-AU"]]
    df["seq_id"] = df["seq_id"].str.replace(r"\s+\d+$", "", regex=True)
    merged_df = pd.merge(df, au_df_subset, on=["seq_id", "dataset"], how="left")
    if only_au:
        merged_df["p-AU_binary"] = merged_df["p-AU"].apply(
            lambda x: 1 if float(x) < 0.05 else 0
        )
    else:
        merged_df["significant_unstable"] = np.where(
            (merged_df["p-AU"] < 0.05) & (merged_df["tii"] != 0),
            1,
            0
        )
    merged_df.drop("p-AU", axis=1, inplace=True)
    df = merged_df
    return df


def balance_datasets(df, only_au=False):
    """
    Subsample rows in df so that we have equal number of stable and unstable rows
    (p-AU<0.05 vs p-AU>=0.05)
    """
    df_list = []
    if only_au:
        col_name = "p-AU_binary"
    else:
        col_name = "significant_unstable"
    # Assuming df is your original DataFrame
    for dataset in pd.unique(df["dataset"]):
        filtered_df = df[df["dataset"] == dataset]
        stable_df = filtered_df[filtered_df[col_name] == 0]
        unstable_df = filtered_df[filtered_df[col_name] == 1]
        num_stable = len(stable_df)
        num_unstable = len(unstable_df)
        # Determine the number to subsample to (the smaller of the two groups)
        num_to_sample = min(num_stable, num_unstable)
        # Subsample from each group
        subsampled_stable = stable_df.sample(n=num_to_sample)
        subsampled_unstable = unstable_df.sample(n=num_to_sample)
        balanced_df = pd.concat([subsampled_stable, subsampled_unstable])
        df_list.append(balanced_df)

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

only_au = False
df = pd.read_csv(combined_statistics, index_col=0)
au_df = pd.read_csv(au_test_results)
df = add_au_test_result(df, au_df, only_au)
df = balance_datasets(df, only_au)
df.to_csv(combined_csv_path)
if only_au:
    model_result = train_random_forest_classifier(
        df, column_name="p-AU_binary", cross_validate=True
    )
else:
    model_result = train_random_forest_classifier(
        df, column_name="significant_unstable", cross_validate=True
    )

model_result.to_csv(output_csv)
