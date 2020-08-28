import numpy as np
import pandas as pd
import re

from datetime import datetime, timedelta
from dateutil import parser
from unidecode import unidecode


def to_normalized_string(original_name):
    ascii_name = to_uppercase_ascii(original_name)
    return re.sub(r'[^\w]+', '_', ascii_name)


def to_uppercase_ascii(unicode_string):
    return unidecode(unicode_string).upper()


def calculate_age(date_of_birth, ref_date, default_ref):
    if pd.isna(date_of_birth):
        return np.nan

    if pd.isna(ref_date):
        ref_date = default_ref

    if isinstance(date_of_birth, str):
        date_of_birth = parser.parse(date_of_birth)

    if isinstance(ref_date, str):
        ref_date = parser.parse(ref_date)

    delta = ref_date - date_of_birth
    age = delta.days // 365

    return max(age, 0)


def calculate_length_of_stay(admission_date, discharge_date, date_of_death):
    if pd.isna(admission_date):
        return np.nan

    if pd.isna(discharge_date) and pd.isna(date_of_death):
        return np.nan

    admission_date = parser.parse(admission_date)
    use_date = discharge_date if pd.notna(discharge_date) else date_of_death
    if isinstance(use_date, str):
        use_date = parser.parse(use_date)

    length_of_stay = use_date - admission_date
    return length_of_stay.days


def count_comorbidities(row):
    if all([pd.isna(v) for v in row.to_numpy()]):
        return np.nan

    comorbidities = sum([1 for v in row.to_numpy() if not pd.isna(v) and bool(v)])
    return comorbidities


def check_death(row):
    death = np.nan

    if pd.notna(row["data_obito"]):
        death = True

    elif pd.notna(row["data_alta_hospitalar"]):
        death = False

    return death


def sum_date_with_interval(ref_date, days):
    if pd.isna(ref_date) or pd.isna(days):
        return np.nan

    if isinstance(ref_date, str):
        ref_date = parser.parse(ref_date)

    interval = timedelta(days=days)
    return ref_date + interval


def estimate_p_healthy_lungs(row):
    # Map the scores to percentages of healthy lungs.

    tc_score_map = {
        0: 1.0,  # 100% of healthy lungs
        1: 0.5,  # 50% of healthy lungs
        2: 0.0,  # 0% of healthy lungs
    }

    # Then we calculate the mean percentage for each of the six parts of the lungs considered.

    esq_sup_score = tc_score_map.get(row["score_tc_esq_sup"], np.nan)
    esq_med_score = tc_score_map.get(row["score_tc_esq_med"], np.nan)
    esq_inf_score = tc_score_map.get(row["score_tc_esq_inf"], np.nan)

    dir_sup_score = tc_score_map.get(row["score_tc_dir_sup"], np.nan)
    dir_med_score = tc_score_map.get(row["score_tc_dir_med"], np.nan)
    dir_inf_score = tc_score_map.get(row["score_tc_dir_inf"], np.nan)

    scores = [
        esq_sup_score, esq_med_score, esq_inf_score,
        dir_sup_score, dir_med_score, dir_inf_score,
    ]

    if any([pd.isna(v) for v in scores]):
        return np.nan

    return sum(scores) / 6
