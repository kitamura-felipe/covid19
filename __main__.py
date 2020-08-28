import numpy as np
import os
import pandas as pd

from copy import deepcopy

from .akf_prediction import akf_prediction, akf_prediction_loio
from .ards_prediction import ards_prediction, ards_prediction_loio
from .mortality_prediction import mortality_prediction, mortality_prediction_loio
from .sepsis_prediction import sepsis_prediction, sepsis_prediction_loio
from .septic_shock_prediction import septic_shock_prediction, septic_shock_prediction_loio
from .ventilation_prediction import ventilation_prediction, ventilation_prediction_loio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FPATH = os.path.join(BASE_DIR, "data", "covid19_internacao.csv")


def collect_df_statistics(df_final):
    stats_cols = [
        "record_id",
        "instituicao",
        "data_nasc",
        "idade",
        "sexo",
        "alta",
        "obito",
        "n_comorbidades"
    ]

    df_stats = df_final[stats_cols].drop_duplicates(subset=["record_id"])

    mean_age = df_stats.idade.mean()
    std_age = df_stats.idade.std()
    gender_counts = df_stats.sexo.value_counts()

    percentiles = list(range(5, 100, 5))
    v_percentiles = np.percentile(df_stats[df_stats.n_comorbidades.notna()].n_comorbidades, percentiles)
    percentiles_str = "\n".join([f"{p_} - {vp}" for p_, vp in zip(percentiles, v_percentiles)])

    df_stats_deaths = df_stats[df_stats["obito"] == 1]
    mean_age_deaths = df_stats_deaths.idade.mean()
    std_age_deaths = df_stats_deaths.idade.std()
    gender_counts_deaths = df_stats_deaths.sexo.value_counts()

    v_percentiles_deaths = np.percentile(df_stats_deaths[df_stats_deaths.n_comorbidades.notna()].n_comorbidades, percentiles)
    percentiles_deaths_str = "\n".join([f"{p_} - {vp}" for p_, vp in zip(percentiles, v_percentiles_deaths)])

    print(f"""
Idade = {'%.1f' % mean_age} +- {'%.1f' % std_age}
P_homem = {'%.1f' % (100*gender_counts['M'] / (gender_counts['M'] + gender_counts['F']))}

{percentiles_str}

Obitos - Idade = {'%.1f' % mean_age_deaths} +- {'%.1f' % std_age_deaths}
Obitos - P_homem = {'%.1f' % (100*gender_counts_deaths['M'] / (gender_counts_deaths['M'] + gender_counts_deaths['F']))}

{percentiles_deaths_str}
""")

    v_percentiles_desde_admissao = np.percentile(df_final.dias_desde_admissao, percentiles)
    percentiles_desde_admissao_str = "\n".join([f"{p_} - {vp}" for p_, vp in zip(percentiles, v_percentiles_desde_admissao)])

    v_percentiles_antes_desfecho = np.percentile(df_final.dias_desde_admissao, percentiles)
    percentiles_antes_desfecho_str = "\n".join([f"{p_} - {vp}" for p_, vp in zip(percentiles, v_percentiles_antes_desfecho)])

    print(f"""
Dias desde admissao:
{percentiles_desde_admissao_str}

Dias antes do desfecho:
{percentiles_antes_desfecho_str}
""")


if __name__ == "__main__":
    df_final = pd.read_csv(DATA_FPATH)

    mortality_prediction(deepcopy(df_final))
    ards_prediction(deepcopy(df_final))
    ventilation_prediction(deepcopy(df_final))
    sepsis_prediction(deepcopy(df_final))
    septic_shock_prediction(deepcopy(df_final))
    akf_prediction(deepcopy(df_final))
    collect_df_statistics(deepcopy(df_final))

    # Leave One Institution Out

    # Leave 9Julho Out
    mortality_prediction_loio(deepcopy(df_final), "9Julho")
    ards_prediction_loio(deepcopy(df_final), "9Julho")
    ventilation_prediction_loio(deepcopy(df_final), "9Julho")
    sepsis_prediction_loio(deepcopy(df_final), "9Julho")
    septic_shock_prediction_loio(deepcopy(df_final), "9Julho")
    akf_prediction_loio(deepcopy(df_final), "9Julho")

    # Leave HAOC Out
    mortality_prediction_loio(deepcopy(df_final), "HAOC")
    ards_prediction_loio(deepcopy(df_final), "HAOC")
    ventilation_prediction_loio(deepcopy(df_final), "HAOC")
    sepsis_prediction_loio(deepcopy(df_final), "HAOC")
    septic_shock_prediction_loio(deepcopy(df_final), "HAOC")
    akf_prediction_loio(deepcopy(df_final), "HAOC")

    # Leave SantaPaula Out
    mortality_prediction_loio(deepcopy(df_final), "SantaPaula")
    ards_prediction_loio(deepcopy(df_final), "SantaPaula")
    ventilation_prediction_loio(deepcopy(df_final), "SantaPaula")
    sepsis_prediction_loio(deepcopy(df_final), "SantaPaula")
    septic_shock_prediction_loio(deepcopy(df_final), "SantaPaula")
    akf_prediction_loio(deepcopy(df_final), "SantaPaula")

    # NOTE: Hospital São Lucas does not have any deaths, so the models do not run properly

    # Leave SaoLucas Out
    # mortality_prediction_loio(deepcopy(df_final), "SaoLucas")
    # ards_prediction_loio(deepcopy(df_final), "SaoLucas")
    # ventilation_prediction_loio(deepcopy(df_final), "SaoLucas")
    # sepsis_prediction_loio(deepcopy(df_final), "SaoLucas")
    # septic_shock_prediction_loio(deepcopy(df_final), "SaoLucas")
    # akf_prediction_loio(deepcopy(df_final), "SaoLucas")

    # Leave UERJ Out
    mortality_prediction_loio(deepcopy(df_final), "UERJ")
    ards_prediction_loio(deepcopy(df_final), "UERJ")
    ventilation_prediction_loio(deepcopy(df_final), "UERJ")
    sepsis_prediction_loio(deepcopy(df_final), "UERJ")
    septic_shock_prediction_loio(deepcopy(df_final), "UERJ")
    akf_prediction_loio(deepcopy(df_final), "UERJ")

    # Leave UFRJ Out
    mortality_prediction_loio(deepcopy(df_final), "UFRJ")
    ards_prediction_loio(deepcopy(df_final), "UFRJ")
    ventilation_prediction_loio(deepcopy(df_final), "UFRJ")
    sepsis_prediction_loio(deepcopy(df_final), "UFRJ")
    septic_shock_prediction_loio(deepcopy(df_final), "UFRJ")
    akf_prediction_loio(deepcopy(df_final), "UFRJ")

    # Leave UNIFESP Out
    mortality_prediction_loio(deepcopy(df_final), "UNIFESP")
    ards_prediction_loio(deepcopy(df_final), "UNIFESP")
    ventilation_prediction_loio(deepcopy(df_final), "UNIFESP")
    sepsis_prediction_loio(deepcopy(df_final), "UNIFESP")
    septic_shock_prediction_loio(deepcopy(df_final), "UNIFESP")
    akf_prediction_loio(deepcopy(df_final), "UNIFESP")
