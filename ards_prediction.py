import os
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split

from .utils.plot import (
    plot_results_1x2,
    plot_results_2x2,
    plot_shap_values,
    plot_survival,
    plot_sensitivity_specificity_vs_threshold
)
from .utils.preprocess import preprocess
from .utils.report import generate_experiment_report
from .run_experiment import run_experiment

pd.options.mode.chained_assignment = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FPATH = os.path.join(BASE_DIR, "data", "covid19_internacao.csv")


def ards_prediction(df_final):
    df_final = preprocess(df_final)

    # Now we expect to prepare our training pipeline

    features_display_names = [
        ("idade", "age"),
        ("seg_normal", "% healthy lungs"),
        ("taxa_gordura", "% fat"),
        ("sofa_score", "sofa score"),
        ("n_comorbidades", "N comorbidities"),
    ]

    features = [
        f[0] for f in features_display_names
    ]

    target = "sdra"

    # We select a small subset of features, and maybe there will be left some duplicates in the dataframe.
    # We drop those duplicates.
    df_model = df_final.drop_duplicates(subset=features + ["record_id", target])

    # Train, validation and test split is in the patient level
    df_split = df_model.groupby("record_id").agg({
        "idade": lambda series: series.iloc[0],
        "sexo_M": lambda series: series.iloc[0],
        "instituicao": lambda series: series.iloc[0],
        target: lambda series: series.iloc[0]
    }).reset_index()

    target_unknown = df_split[df_split[target].isna()].record_id.nunique()
    df_split = df_split.dropna(subset=[target])

    train_valid_records, test_records = train_test_split(
        df_split, test_size=0.2, random_state=0, stratify=df_split[target]
    )

    assert len(set(train_valid_records.record_id.unique()) & set(test_records.record_id.unique())) == 0

    summaries, df_test = run_experiment(df_model, train_valid_records, test_records, features, target)
    X_test = df_test[features]

    ############################## Finished training the models ##############################

    save_path_2x2 = os.path.join(BASE_DIR, "desfechos_intermediarios", "ards_model_2x2.pdf")
    plot_results_2x2(summaries, save_path_2x2)

    save_path_1x2 = os.path.join(BASE_DIR, "desfechos_intermediarios", "ards_model_1x2.pdf")
    metrics_summary = plot_results_1x2(summaries, save_path_1x2)

    save_path_shap = os.path.join(BASE_DIR, "desfechos_intermediarios", "ards_shap.pdf")
    shap_values_plot = plot_shap_values(X_test, summaries, [f[1] for f in features_display_names], save_path_shap)

    save_path_sens_spec = os.path.join(BASE_DIR, "desfechos_intermediarios", "ards_sens_spec.pdf")
    plot_sensitivity_specificity_vs_threshold(summaries, save_path_sens_spec)

    save_report = os.path.join(BASE_DIR, "desfechos_intermediarios", "ards_report.txt")
    reports = generate_experiment_report(
        target, df_split, df_final, features, metrics_summary,
        train_valid_records, test_records, save_report
    )

    print(reports["stats_report"])
    print(reports["missing_values"])
    print(reports["metrics"])

    ############################## Survival analysis ##############################

    save_path_survival = os.path.join(BASE_DIR, "desfechos_intermediarios", "ards_survival.pdf")
    plot_survival(df_test, features, summaries, save_path_survival)


def ards_prediction_loio(df_final, institution):
    """ Leave one institution out """
    df_final = preprocess(df_final)

    # The same institution might appear with different names, so we make a list with the names
    assert isinstance(institution, str) or isinstance(institution, list), "'institution' must be either a string or a list"
    if isinstance(institution, str):
        institution = [institution]

    # Now we expect to prepare our training pipeline

    features_display_names = [
        ("idade", "age"),
        ("seg_normal", "% healthy lungs"),
        ("taxa_gordura", "% fat"),
        ("sofa_score", "sofa score"),
        ("n_comorbidades", "N comorbidities"),
    ]

    features = [
        f[0] for f in features_display_names
    ]

    target = "sdra"

    # We select a small subset of features, and maybe there will be left some duplicates in the dataframe.
    # We drop those duplicates.
    df_model = df_final.drop_duplicates(subset=features + ["record_id", target])

    # Train, validation and test split is in the patient level
    df_split = df_model.groupby("record_id").agg({
        "idade": lambda series: series.iloc[0],
        "sexo_M": lambda series: series.iloc[0],
        "instituicao": lambda series: series.iloc[0],
        target: lambda series: series.iloc[0]
    }).reset_index()

    target_unknown = df_split[df_split[target].isna()].record_id.nunique()
    df_split = df_split.dropna(subset=[target])

    # Leave institution out of the train/validation pipeline
    train_valid_records = df_split[~df_split.instituicao.isin(institution)]
    test_records = df_split[df_split.instituicao.isin(institution)]

    assert len(set(train_valid_records.record_id.unique()) & set(test_records.record_id.unique())) == 0

    summaries, df_test = run_experiment(df_model, train_valid_records, test_records, features, target)
    X_test = df_test[features]

    ############################## Finished training the models ##############################

    save_path_2x2 = os.path.join(BASE_DIR, "desfechos_intermediarios", "LOIO", f"{institution[0]}_ards_model_2x2.pdf")
    plot_results_2x2(summaries, save_path_2x2)

    save_path_1x2 = os.path.join(BASE_DIR, "desfechos_intermediarios", "LOIO", f"{institution[0]}_ards_model_1x2.pdf")
    metrics_summary = plot_results_1x2(summaries, save_path_1x2)

    save_path_shap = os.path.join(BASE_DIR, "desfechos_intermediarios", "LOIO", f"{institution[0]}_ards_shap.pdf")
    shap_values_plot = plot_shap_values(X_test, summaries, [f[1] for f in features_display_names], save_path_shap)

    save_path_sens_spec = os.path.join(BASE_DIR, "desfechos_intermediarios", "LOIO", f"{institution[0]}_ards_sens_spec.pdf")
    plot_sensitivity_specificity_vs_threshold(summaries, save_path_sens_spec)

    save_report = os.path.join(BASE_DIR, "desfechos_intermediarios", "LOIO", f"{institution[0]}_ards_report.txt")
    reports = generate_experiment_report(
        "ARDS", target, df_split, df_final, features, metrics_summary,
        train_valid_records, test_records, save_report
    )

    print(reports["stats_report"])
    print(reports["missing_values"])
    print(reports["metrics"])

    ############################## Survival analysis ##############################

    # save_path_survival = os.path.join(BASE_DIR, "desfechos_intermediarios", "LOIO", f"{institution[0]}_ards_survival.pdf")
    # plot_survival(df_test, features, summaries, save_path_survival)

if __name__ == "__main__":
    df_final = pd.read_csv(DATA_FPATH)
    mortality_prediction(df_final)
