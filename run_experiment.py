import numpy as np

from catboost import CatBoostClassifier, Pool
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score,
                             roc_curve,
                             precision_recall_curve,
                             average_precision_score,
                             f1_score,
                             accuracy_score)


def run_experiment(df_model, train_valid_records, test_records, features, target):
    df_test = df_model[df_model.record_id.isin(test_records.record_id)]
    X_test = df_test[features]
    y_test = df_test[target].map({False: 0, True: 1})

    summaries = []
    skf = StratifiedKFold(n_splits=5)
    for i, (train, valid) in enumerate(skf.split(train_valid_records.record_id, train_valid_records[target])):
        train_records = train_valid_records.iloc[train].record_id
        df_train = df_model[df_model.record_id.isin(train_records)]

        valid_records = train_valid_records.iloc[valid].record_id
        df_valid = df_model[df_model.record_id.isin(valid_records)]

        assert len(set(df_train.record_id.unique()) & set(df_valid.record_id.unique())) == 0
        assert len(set(df_train.record_id.unique()) & set(df_test.record_id.unique())) == 0
        assert len(set(df_test.record_id.unique()) & set(df_valid.record_id.unique())) == 0

        model = CatBoostClassifier(max_depth=3, learning_rate=0.01, early_stopping_rounds=100)

        X_train = df_train[features]
        y_train = df_train[target].map({False: 0, True: 1})

        X_valid = df_valid[features]
        y_valid = df_valid[target].map({False: 0, True: 1})

        valid_counts = y_valid.value_counts()

        model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
        pred = model.predict_proba(X_test)
        y_score = pred[:, 1]

        auc = roc_auc_score(y_test.values, y_score)
        fpr, tpr, _ = roc_curve(y_test.values, y_score)

        auc = roc_auc_score(y_test.values, y_score)
        fpr, tpr, thresholds = roc_curve(y_test.values, y_score)

        ap = average_precision_score(y_test.values, y_score)
        precision, recall, _ = precision_recall_curve(y_test.values, y_score)

        f1s = [(thr, f1_score(y_test.values, y_score > thr)) for thr in np.arange(0, 1, 0.01)]
        accs = [(thr, accuracy_score(y_test.values, y_score > thr)) for thr in np.arange(0, 1, 0.01)]
        best_thr_f1, best_f1 = max(f1s, key=lambda t: t[1])
        best_thr_acc, best_acc = max(accs, key=lambda t: t[1])
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test.values, y_score, n_bins=7)

        shap_values = model.get_feature_importance(Pool(X_test, y_test), type="ShapValues")
        expected_value = shap_values[0, -1]
        shap_values = shap_values[:, :-1]

        summary = {
            "model": model,
            "roc_auc": auc,
            "avg_precision": ap,
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "precision": precision,
            "recall": recall,
            "best_thr_f1": best_thr_f1,
            "best_f1": best_f1,
            "best_thr_acc": best_thr_acc,
            "best_acc": best_acc,
            "fraction_of_positives": fraction_of_positives,
            "mean_predicted_value": mean_predicted_value,
            "expected_value": expected_value,
            "shap_values": shap_values
        }

        summaries.append(summary)

        print(f"""
--------------------------------------------------------------------
Finished pipeline for fold #{i + 1}

Summary:

Positive cases : {valid_counts[1]}
Negative cases : {valid_counts[0]}

ROC AUC = {auc}
Average Precision = {ap}
Best F1 = {best_f1} (Threshold = {best_thr_f1})
Best Accuracy = {best_acc} (Threshold = {best_thr_acc})
--------------------------------------------------------------------
""")

    return summaries, df_test
