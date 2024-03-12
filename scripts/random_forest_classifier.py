def random_forest_classification(column_name, combined_csv_path, output_csv, model_features_csv, classifier_metrics_csv, parameter_file, data_folder):
    import optuna
    import pandas as pd
    import numpy as np
    import json
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_curve, f1_score

    # taxon_name_col = "seq_id"
    cols_to_drop = [
        "seq_id",
        "likelihood",
        "normalised_tii",
        "dataset",
        "rf_radius",
        "change_to_low_bootstrap_dist",
        "tii",
    ]


    def train_random_forest_classifier(df, column_name):
        X = df.drop(cols_to_drop, axis=1)
        y = df[column_name]
        X = X.drop([column_name], axis=1)

        # Split data into training+validation and test sets
        try:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
                # Note: test_size=0.25 in this split will actually result in 20% of the original data being set aside for validation,
                # because 0.25 * 0.8 (remaining after first split) = 0.2
            )
        except ValueError as e:
            print(e)
            return(False)

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
            min_samples_split = trial.suggest_float("min_samples_split", 0.00001, 1, log=True)
            min_samples_leaf = trial.suggest_float("min_samples_leaf", 0.00001, 1, log=True)
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
            model.fit(X_val_imputed, y_val)
            y_pred = model.predict(X_val_imputed)
            f1 = f1_score(y_val, y_pred)
            return f1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=200)
        best_params = study.best_params
        with open(parameter_file, "w") as f:
            json.dump(best_params, f, indent=4)


        # Train model on num_classifier balanced training sets
        if len(y_train[y_train == 1]) < len(y_train[y_train ==0]):
            X_small = X_train_imputed_df[y_train == 1]
            X_large = X_train_imputed_df[y_train == 0]
            y_small = y_train[y_train == 1]
            y_large = y_train[y_train == 0]
        else:
            X_small = X_train_imputed_df[y_train == 0]
            X_large = X_train_imputed_df[y_train == 1]
            y_small = y_train[y_train == 0]
            y_large = y_train[y_train == 1]
        # we assume that y_unstable is shorter than y_stable!
        subset_size = len(y_small) # for balancing
        classifiers = []
        X_balanced_sets = []
        y_balanced_sets = []
        num_classifiers = int(len(y_large)/len(y_small))
        print("Total number of classifiers: ", num_classifiers)
        for i in range(num_classifiers):
            print("Training classifier number ", i)
            # Randomly sample without replacement using pandas
            X_large_subset = X_large.sample(n=subset_size, replace=False, random_state=42)
            y_large_subset = y_large.loc[X_large_subset.index]

            # Combine with the unstable samples
            X_balanced = pd.concat([X_small, X_large_subset])
            y_balanced = pd.concat([y_small, y_large_subset])

            X_balanced_sets.append(X_balanced)
            y_balanced_sets.append(y_balanced)

            # Fit the model
            fit_model = RandomForestClassifier(**best_params, random_state=42)
            fit_model.fit(X_balanced, y_balanced)
            classifiers.append(fit_model)

            # Drop the sampled rows
            X_large = X_large.drop(X_large_subset.index)
            y_large = y_large.drop(X_large_subset.index)

        # Save all balanced training sets to files
        for i, (X_balanced, y_balanced) in enumerate(zip(X_balanced_sets, y_balanced_sets)):
            y_balanced_df = pd.DataFrame(y_balanced, columns=[column_name])
            combined_df = pd.concat([X_balanced, y_balanced_df], axis=1)
            combined_df.to_csv(data_folder + "balanced_training_set_" + column_name +"_classifier_" + str(i) +".csv", index=False)

        # Balance test set
        # Combine X_test and y_test into a single DataFrame for easy subsampling
        y_test_df = pd.DataFrame(y_test, columns=[column_name]).reset_index(drop=True)
        X_test_df = pd.concat([X_test_imputed_df.reset_index(drop=True), y_test_df], axis=1)
        group_0 = X_test_df[X_test_df[column_name] == 0]
        group_not_0 = X_test_df[X_test_df[column_name] != 0]
        min_size = min(len(group_0), len(group_not_0))
        subsampled_group_0 = group_0.sample(n=min_size, random_state=42)
        subsampled_group_not_0 = group_not_0.sample(n=min_size, random_state=42)
        balanced_test_df = pd.concat([subsampled_group_0, subsampled_group_not_0]).sample(frac=1, random_state=42).reset_index(drop=True)

        # Split the balanced DataFrame back into X and y components
        X_test_balanced = balanced_test_df.drop(column_name, axis=1)
        y_test_balanced = balanced_test_df[column_name]
        
        # Run model on test set
        predictions = np.array([clf.predict(X_test_balanced) for clf in classifiers])
        model_results = pd.DataFrame(
            {
                "actual": y_test_balanced
            }
        )
        pred_col = {f'prediction{i}': predictions[i] for i in range(len(predictions))}
        model_results = pd.concat([model_results, pd.DataFrame(pred_col)], axis=1)

        probabilities = np.array([clf.predict_proba(X_test_balanced)[:, 1] for clf in classifiers])
        average_probabilities = np.mean(probabilities, axis=0)

        # Compute ROC curve and AUC for the average probabilities
        fpr, tpr, _ = roc_curve(y_test_balanced, average_probabilities)
        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(classifier_metrics_csv)

        # Aggregating feature importances
        feature_importances = np.array([clf.feature_importances_ for clf in classifiers])
        avg_importances = np.mean(feature_importances, axis=0)
        feature_names = X_train_imputed_df.columns
        pd.DataFrame({"model_importance": avg_importances}, index=feature_names).to_csv(model_features_csv)

        model_importances = [clf.feature_importances_ for clf in classifiers]
        for i in range(len(model_importances)):
            pd.DataFrame({"model_importance": model_importances[i]}, index=feature_names).to_csv(data_folder+"model_features_" + column_name + "_classifier_" + str(i) + ".csv")
        return model_results


    df = pd.read_csv(combined_csv_path, index_col=0)
    if column_name == "significant_unstable":
        def add_au_test_result(df, au_df, only_au=False):
            """
            Add results from au_test in given file au_test_results to df containing all summary statistics
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
                    (merged_df["p-AU"] < 0.05) & (merged_df["tii"] != 0), 1, 0
                )
            merged_df.drop("p-AU", axis=1, inplace=True)
            df = merged_df
            return df

        only_au = False
        au_test_results = data_folder+"au_test_result.csv"
        au_df = pd.read_csv(au_test_results)
        df = add_au_test_result(df, au_df, only_au)
        df.to_csv(data_folder + "au_test_combined_statistics.csv")
        if only_au:
            model_result = train_random_forest_classifier(
                df, column_name="p-AU_binary"
            )
        else:
            model_result = train_random_forest_classifier(
                df, column_name="significant_unstable"
            )
    else: # classifying stability (TII=0 vs !=0)
        df[column_name + "_binary"] = [1 if x > 0 else 0 for x in df[column_name]]
        column_name = column_name + "_binary"
        model_result = train_random_forest_classifier(df, column_name)
        
    if not isinstance(model_result, pd.DataFrame):
        with open(output_csv, "w") as f:
            pass
        with open(model_features_csv, "w") as f:
            pass
        with open(classifier_metrics_csv, "w") as f:
            pass
        raise ValueError(f"Not enough samples for classification of significant instability to balance training set.")

    model_result.to_csv(output_csv)