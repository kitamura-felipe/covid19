import funcy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

from dateutil import parser
from tqdm import tqdm

from utils.helpers import *
from utils.plot import plot_joint_distribution

font = {
    "size": 30
}
matplotlib.rc("font", **font)

pd.options.mode.chained_assignment = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MOST_RECENT_FILE = sorted(os.listdir(os.path.join(BASE_DIR, "data", "REDCap")))[-1]
REDCAP_FPATH = os.path.join(BASE_DIR, "data", "REDCap", MOST_RECENT_FILE)
SERIES_ID_FPATH = os.path.join(BASE_DIR, "data", "match_redcap_plataforma.csv")
SEGMENTATION_FPATH = os.path.join(BASE_DIR, "data", "inference_df.csv")

get_date_regex = r"ProjetoCOVIDAI_DATA_(?P<data>.*)_\d+.csv"
date_str = re.match(get_date_regex, MOST_RECENT_FILE).group("data")
dataset_date = parser.parse(date_str)

# Normalize name and CPF

df = pd.read_csv(REDCAP_FPATH)
df.nome = df.nome.apply(lambda s: to_normalized_string(s) if pd.notna(s) else s)
df.cpf = df.cpf.apply(lambda v: str(int(v)) if pd.notna(v) else v)

# Fill redcap_repeat_instrument missing data with "dados_pessoais_unico" since these
# rows are not filled automatically by the database

df.redcap_repeat_instrument = df.redcap_repeat_instrument.fillna("dados_pessoais_unico")

# Fill the missing hospitalization date with date of admission to ICU if existent

df.data_admissao_hospitalar = df.data_admissao_hospitalar.fillna(df.data_admissao_uti)

# Calculate length of stay based on hospitalization date and date of discharge or
# date of death

fill_length_of_stay = df.apply(
    lambda row: calculate_length_of_stay(
        row["data_admissao_hospitalar"],
        row["data_alta_hospitalar"],
        row["data_obito"]
    ),
    axis=1
)

df.tempo_estadia_hospitalar = df.tempo_estadia_hospitalar.fillna(fill_length_of_stay)

# Calculate the date of discharge from ICU based on the date of admission
# in the ICU and length of stay in the ICU.

df["data_alta_uti"] = df.apply(
    lambda row: sum_date_with_interval(
        row["data_admissao_uti"],
        row["tempo_estadia_uti"]
    ),
    axis=1
)

# Calculate the date of removal of the ventilation based on the date of ventilation
# and the length of ventilation

df["data_remocao_ventilacao"] = df.apply(
    lambda row: sum_date_with_interval(
        row["data_ventilacao"],
        row["tempo_ventilacao_mecanica"]
    ),
    axis=1
)

# Calculate age and body mass index

df["idade"] = df.apply(
    lambda row: calculate_age(
        row["data_nasc"],
        row["data_admissao_hospitalar"],
        dataset_date
    ),
    axis=1
)

df["imc"] = df.peso / (df.altura ** 2)

# Some of the rows have the plaquets number in a different unity and need to be
# multiplied by 1000

df.plaquetas = df.plaquetas.apply(lambda v: v * 1000 if v < 1000 else v)

############################## Finished processing the ordinary data ##############################

# Here we define variables useful for processing the rest of the data

cols_intermediate_outcomes = [
    "data_sepse",
    "sepse",
    "data_sdra",
    "sdra",
    "data_falencia_cardiaca",
    "falencia_cardiaca",
    "data_choque_septico",
    "choque_septico",
    "data_coagulopatia",
    "coagulopatia",
    "data_iam",
    "iam",
    "data_ira",
    "ira"
]

cols_personal_data = [
    "nome",
    "cpf",
    "instituicao",
    "data_nasc",
    "idade",
    "sexo",
    "altura",
    "peso",
    "imc",
    "alta",
    "obito",
    "data_admissao_hospitalar",
    "data_admissao_uti",
    "data_obito",
    "data_alta_hospitalar",
    "data_alta_uti",
    "data_ventilacao",
    "data_remocao_ventilacao",
    "tempo_estadia_hospitalar",
    "tempo_estadia_uti",
    "tempo_ventilacao_mecanica"
] + cols_intermediate_outcomes

cols_comorbidities = [
    "has",
    "ieca_bra",
    "dm",
    "asma",
    "tabagista",
    "dpoc",
    "cardiopatia",
    "irc",
    "neoplasia",
    "aids",
    "neutropenia"
]

cols_respiratory_comorbidities = [
    "asma", "tabagista", "dpoc"
]

cols_cardiac_comorbidities = [
    "has", "cardiopatia"
]

cols_dates = [
    col for col in df.columns
    if "data" in col and col not in
    cols_personal_data + ["redcap_data_access_group"]
]

identity_map = {
    0: 0,
    1: 1
}

irc_map = {
    1: "negativo",
    2: "nao_dialitico",
    3: "dialitico"
}

neoplasia_map = {
    1: "negativo",
    2: "primaria_ou_secundaria",
    3: "outras"
}

map_comorbidities = {
    "irc": irc_map,
    "neoplasia": neoplasia_map
}

# Now we build a separate dataframe for saving pesonal data.

df_personal_data = df[df.redcap_repeat_instrument == "dados_pessoais_unico"]

# Discriminate patients that were admitted to the hospital and to the ICU. Also, discriminate those that
# were discharged and those who died.

df_personal_data["internacao"] = df_personal_data.data_admissao_hospitalar.notna()
df_personal_data["uti"] = df_personal_data.data_admissao_uti.notna()
df_personal_data["obito"] = df_personal_data.data_obito.notna()
df_personal_data["alta"] = df_personal_data.data_alta_hospitalar.notna()

df_personal_data = df_personal_data[
    ["record_id"] + cols_personal_data + cols_comorbidities
]

for col in cols_comorbidities:
    df_personal_data[col] = df_personal_data[col].map(map_comorbidities.get(col, identity_map))

# Count the number of previous comorbidities each patient has.

df_personal_data["n_comorbidades"] = df_personal_data[cols_comorbidities].apply(count_comorbidities, axis=1)
df_personal_data["n_comorbidades_respiratorias"] = df_personal_data[cols_respiratory_comorbidities].apply(count_comorbidities, axis=1)
df_personal_data["n_comorbidades_cardiacas"] = df_personal_data[cols_cardiac_comorbidities].apply(count_comorbidities, axis=1)

############################## Finished processing the personal data ##############################

# Now we build separate dataframes for saving clinical, treatment, laboratorial, image and confirmatory data.

# Clinical dataframe

cols_clinical = [
    "data_dispneia",
    "dispneia",
    "data_sofa",
    "sofa_score",
    "data_saturacao_o2",
    "saturacao_o2",
    "data_saps_3",
    "saps_3"
]

df_clinical = df[df.redcap_repeat_instrument == "evolucao_clinica_multiplo"]
df_clinical = df_clinical[["record_id"] + cols_clinical]

# We need separate dataframes for each date. Note that the clinical dataframe has four date. We will separate
# the columns accordingly.

df_dispneia = df_clinical[[
    "record_id",
    "data_dispneia",
    "dispneia"
]]

df_sofa = df_clinical[[
    "record_id",
    "data_sofa",
    "sofa_score"
]]

df_saturacao_o2 = df_clinical[[
    "record_id",
    "data_saturacao_o2",
    "saturacao_o2"
]]

df_saps_3 = df_clinical[[
    "record_id",
    "data_saps_3",
    "saps_3"
]]

# Treatment dataframe

cols_treatment = [
    "data_ventilacao",
    "ventilacao",
    "pao2_fio2",
    "data_pronacao",
    "pronacao",
    "data_hemodialise",
    "hemodialise"
]

df_treatment = df[df.redcap_repeat_instrument == "evolucao_tratamento_multiplo"]
df_treatment = df_treatment[["record_id"] + cols_treatment]

# Note that the treatment dataframe has four date. We will separate the columns accordingly
# just as we did for the clinical dataframe.

df_ventilacao = df_treatment[[
    "record_id",
    "data_ventilacao",
    "ventilacao",
    "pao2_fio2"
]]

df_pronacao = df_treatment[[
    "record_id",
    "data_pronacao",
    "pronacao"
]]

df_hemodialise = df_treatment[[
    "record_id" ,
    "data_hemodialise",
    "hemodialise"
]]

# Laboratory results dataframe

cols_laboratory = [
    "leucocitos",
    "linfocitos",
    "neutrofilos",
    "tgp",
    "creatinina",
    "pcr",
    "d_dimero",
    "il_6",
    "plaquetas",
    "rni",
    "troponina",
    "pro_bnp",
    "bicarbonato",
    "lactato"
]

df_laboratory = df[df.redcap_repeat_instrument == "evolucao_laboratorial_multiplo"]
df_laboratory = df_laboratory[["record_id", "data_resultados_lab"] + cols_laboratory]

# Image dataframe

cols_image = [
    "uid_imagem",
    "tipo_imagem",
    "data_imagem",
    "padrao_imagem_rsna",
    "score_tc_dir_sup",
    "score_tc_dir_med",
    "score_tc_dir_inf",
    "score_tc_esq_sup",
    "score_tc_esq_med",
    "score_tc_esq_inf"
]

df_image = df[df.redcap_repeat_instrument == "evolucao_imagem_multiplo"]
df_image.uid_imagem = df_image.uid_imagem.apply(lambda s: s.strip() if pd.notna(s) else s)
df_image = df_image[["record_id", "redcap_repeat_instance"] + cols_image]

df_image = pd.merge(
    left=df_personal_data[["record_id", "nome", "data_nasc", "data_admissao_hospitalar", "instituicao"]],
    right=df_image,
    how="right",
    on="record_id",
    validate="one_to_many"
)

uids_internados = set(df_image[df_image.data_admissao_hospitalar.notna()].uid_imagem.unique())

# For images, we also have the data retrieved from the deep segmentation model. We need
# to enrich our dataframe with the percentage of healthy lungs, affected by ground-glass opacity
# and consolidation, and the amount of fat in patient's body.

cols_series_id = [
    "record_id",
    "redcap_repeat_instance",
    "infer_series_id"
]

df_series_id = pd.read_csv(SERIES_ID_FPATH, sep=";")
df_series_id = df_series_id[cols_series_id]
df_series_id = df_series_id.drop_duplicates()

cols_segmentation = [
    "UID_Plataforma",
    "series_id",
    "seg_consolidacao",
    "seg_normal",
    "seg_vf1",
    "seg_vf2",
    "seg_vf3",
    "volume_pulmao",
    "taxa_gordura",
    "volume_gordura",
    "mediastino"
]

tmp_data = []
df_seg_raw = pd.read_csv(SEGMENTATION_FPATH)
df_seg_raw = df_seg_raw[cols_segmentation]
df_seg_raw = df_seg_raw[df_seg_raw.volume_pulmao >= 1.]
df_seg_raw = pd.merge(left=df_series_id, right=df_seg_raw, left_on="infer_series_id", right_on="series_id", how="right")

# Each TC study might have multiple series. We need to select the one with

grouped = df_seg_raw.groupby("UID_Plataforma")
for uid_imagem, group in grouped:
    if any(group.mediastino):
        use_group = group[group.mediastino]
    else:
        use_group = group

    sorted_group = use_group.sort_values("volume_pulmao")
    tmp_data.append(
        dict(sorted_group.iloc[-1])
    )

df_seg = pd.DataFrame(tmp_data)
df_seg = df_seg[df_seg.seg_normal.notna()]

df_image = pd.merge(
    left=df_image,
    right=df_seg,
    how="left",
    on=["record_id", "redcap_repeat_instance"]
)

df_image[
    ["record_id", "redcap_repeat_instance", "nome", "data_nasc", "data_admissao_hospitalar", "instituicao"] + cols_image
].to_csv(os.path.join(BASE_DIR, "data", "TC_scans.csv"), index=False)
df_image = df_image.rename({"redcap_repeat_instance": "redcap_repeat_instance_image"})

df_matches = df_image[
    (df_image.seg_normal.notna()) & (df_image.data_admissao_hospitalar.notna())
]

df_matches[
    ["record_id", "data_admissao_hospitalar", "instituicao", "data_imagem", "uid_imagem"]
].to_csv(os.path.join(BASE_DIR, "data", "matches.csv"), index=False)

n_matches = df_matches.uid_imagem.nunique()

print(f"{n_matches} between REDCap and segmentation\n")

# COVID-19 confirmation dataframe

df_confirmation = df[df.redcap_repeat_instrument == "confirmacao_covid_multiplo"]

############################## Finished processing the results data ##############################

# Now we are going to create a dataframe that each row corresponds to a moment in the patient stay at the
# hospital. For each date in the patient history, we will update the row with the latest information about
# that patient.

# First, we need to define some helper functions to work on the processing of the data.

def get_group(grouped, key, default_columns):
    """
    Gets a group by key from a Pandas Group By object. If the key does not exist, returns an empty
    group with the default columns.
    """

    if key in grouped.groups:
        group = grouped.get_group(key)
    else:
        group = pd.DataFrame([], columns=default_columns)

    return group


def last_register_before_date(registers, date_col, date, default_columns):
    """
    Gets the last register before a reference date in a dataframe. If there are no register before the
    date, returns an empty register with the default columns.
    """

    registers = registers[registers[date_col].notna()]
    registers_before_date = registers[
        registers[date_col].apply(parser.parse) <= date
    ]
    if len(registers_before_date) == 0:
        registers_before_date = pd.DataFrame([(np.nan for col in default_columns)], columns=default_columns)

    last_register = registers_before_date.iloc[-1]
    return last_register

# Theb, we need to group by patient all the dataframes we built previously.

grouped_dispneia = df_dispneia.groupby("record_id")
grouped_sofa = df_sofa.groupby("record_id")
grouped_saturacao_o2 = df_saturacao_o2.groupby("record_id")
grouped_saps_3 = df_saps_3.groupby("record_id")
grouped_image = df_image.groupby("record_id")
grouped_laboratory = df_laboratory.groupby("record_id")
grouped_ventilacao = df_ventilacao.groupby("record_id")
grouped_pronacao = df_pronacao.groupby("record_id")
grouped_hemodialise = df_hemodialise.groupby("record_id")

# Now we iterate over the personal data dataframe, which has one row per patient.

after_discharge = []
after_death = []
new_rows = []

for i, row in tqdm(df_personal_data.iterrows(), total=len(df_personal_data)):
    record_id = row["record_id"]
    institution = row["instituicao"]

    hospitalization_date = row["data_admissao_hospitalar"]
    discharge_date = row["data_alta_hospitalar"]
    date_of_death = row["data_obito"]

    if pd.notna(date_of_death):
        date_of_death = parser.parse(date_of_death)
    if pd.notna(discharge_date):
        discharge_date = parser.parse(discharge_date)
    if pd.notna(hospitalization_date):
        hospitalization_date = parser.parse(hospitalization_date)

    # Get each group and sort by the date

    group_dispneia = get_group(
        grouped_dispneia, record_id, df_dispneia.columns
    ).sort_values("data_dispneia")

    group_sofa = get_group(
        grouped_sofa, record_id, df_sofa.columns
    )

    group_saturacao_o2 = get_group(
        grouped_saturacao_o2, record_id, df_saturacao_o2.columns
    )

    group_saps_3 = get_group(
        grouped_saps_3, record_id, df_saps_3.columns
    )

    group_image = get_group(
        grouped_image, record_id, df_image.columns
    )

    group_laboratory = get_group(
        grouped_laboratory, record_id, df_laboratory.columns
    )

    group_ventilacao = get_group(
        grouped_ventilacao, record_id, df_ventilacao.columns
    )

    group_pronacao = get_group(
        grouped_pronacao, record_id, df_pronacao.columns
    )

    group_hemodialise = get_group(
        grouped_hemodialise, record_id, df_hemodialise.columns
    )

    # List the dates available for the patient

    patient_dates = set(filter(
        pd.notna,
        list(group_dispneia.data_dispneia) +
        list(group_sofa.data_sofa) +
        list(group_saturacao_o2.data_saturacao_o2) +
        list(group_saps_3.data_saps_3) +
        list(group_image.data_imagem) +
        list(group_laboratory.data_resultados_lab) +
        list(group_ventilacao.data_ventilacao) +
        list(group_pronacao.data_pronacao) +
        list(group_hemodialise.data_hemodialise)
    ))

    patient_dates = funcy.lmap(parser.parse, patient_dates)

    # Now we iterate over the dates of the patient retrieving the last register for
    # each group.

    new_patient_rows = []
    for date_tmp in patient_dates:
        # If the date is after the patient's death or the patient's discharge, we want to ignore
        # the register.

        if abs(date_tmp.year - dataset_date.year) > 0:
            continue

        if pd.notna(date_of_death) and date_tmp > date_of_death:
            after_death.append(record_id)
            continue

        if pd.notna(discharge_date) and date_tmp > discharge_date:
            after_discharge.append(discharge_date)
            continue

        last_register_dispneia = last_register_before_date(group_dispneia, "data_dispneia", date_tmp, df_dispneia.columns)
        last_register_sofa = last_register_before_date(group_sofa, "data_sofa", date_tmp, df_sofa.columns)
        last_register_saturacao_o2 = last_register_before_date(group_saturacao_o2, "data_saturacao_o2", date_tmp, df_saturacao_o2.columns)
        last_register_saps_3 = last_register_before_date(group_saps_3, "data_saps_3", date_tmp, df_saps_3.columns)
        last_register_image = last_register_before_date(group_image, "data_imagem", date_tmp, df_image.columns)
        last_register_laboratory = last_register_before_date(group_laboratory, "data_resultados_lab", date_tmp, df_laboratory.columns)
        last_register_pronacao = last_register_before_date(group_pronacao, "data_pronacao", date_tmp, df_pronacao.columns)
        last_register_hemodialise = last_register_before_date(group_hemodialise, "data_hemodialise", date_tmp, df_hemodialise.columns)

        # Need for mechanical ventilation is one of our target variables. Thus, we do not want to get the last register before the
        # current date. We want to know if the patient ever needed mechanical ventilation at any point in time.

        ventilacao = group_ventilacao[group_ventilacao.ventilacao == group_ventilacao.ventilacao.max()].sort_values("data_ventilacao", ascending=False)
        if len(ventilacao) == 0:
            ventilacao = pd.DataFrame([(np.nan for col in group_ventilacao.columns)], columns=group_ventilacao.columns)
        ventilacao = ventilacao.iloc[-1]

        new_row = {}
        new_row.update(row)
        new_row.update(dict(last_register_dispneia))
        new_row.update(dict(last_register_sofa))
        new_row.update(dict(last_register_saturacao_o2))
        new_row.update(dict(last_register_saps_3))
        new_row.update(dict(last_register_image))
        new_row.update(dict(last_register_laboratory))
        new_row.update(dict(last_register_pronacao))
        new_row.update(dict(last_register_hemodialise))
        new_row.update(dict(ventilacao))
        new_row["data"] = date_tmp
        new_row["record_id"] = record_id
        new_row["instituicao"] = institution

        new_row["dias_desde_admissao"] = (date_tmp - hospitalization_date).days if pd.notna(hospitalization_date) else np.nan
        date_of_outcome = date_of_death if pd.notna(date_of_death) else discharge_date
        new_row["dias_antes_desfecho"] = (date_of_outcome - date_tmp).days if pd.notna(date_of_outcome) else np.nan

        new_patient_rows.append(new_row)
    new_rows.extend(new_patient_rows)

df_final = pd.DataFrame(new_rows)

# We need to calculate some dummy variables for the categorical data.

padrao_rsna_dummies = pd.get_dummies(df_final.padrao_imagem_rsna, prefix="padrao_rsna")
ventilacao_dummies = pd.get_dummies(df_final.ventilacao, prefix="ventilacao")
neoplasia_dummies = pd.get_dummies(df_final.neoplasia, prefix="neoplasia")
irc_dummies = pd.get_dummies(df_final.irc, prefix="irc")
sexo_dummies = pd.get_dummies(df_final.sexo, prefix="sexo")

df_final = pd.concat([df_final,
                      padrao_rsna_dummies,
                      ventilacao_dummies,
                      neoplasia_dummies,
                      irc_dummies,
                      sexo_dummies], axis=1)

def calc_ventilation(row):
    if pd.isna(row["ventilacao"]):
        return row["ventilacao"]

    return row["ventilacao_5.0"] or row["ventilacao_6.0"]

df_final["mechanical_ventilation"] = df_final.apply(calc_ventilation, axis=1)
# df_final.to_csv(os.path.join(BASE_DIR, "data", "covid19_final.csv"), index=False)

# And we want to have a separate file that includes only the data of patients that were hospitalized.

df_internacao = df_final[df_final.data_admissao_hospitalar.notna()].reset_index()
df_internacao.to_csv(os.path.join(BASE_DIR, "data", "covid19_internacao.csv"), index=False)

############################## Statistics ##############################

potentially_elegible = df_final.record_id.nunique()
elegible = df_internacao.record_id.nunique()
still_hospitalized = df_internacao[
    (df_internacao.data_alta_hospitalar.isna()) & (df_internacao.data_obito.isna())
].record_id.nunique()

print(f"""
Potentially elegible participants = {potentially_elegible}
Elegible participants = {elegible}

Excluded (not hospitalized) = {potentially_elegible - elegible}

Index test = {elegible - still_hospitalized}

Excluded (still hospitalized) = {still_hospitalized}
""")

#################################### Plot joint distributions ######################################

save_path_joint = os.path.join(BASE_DIR, "desfechos_finais", "joint_normal_lungs_age.tiff")
plot_joint_distribution(df_internacao, save_path_joint, fformat="tiff")

################################### Segmentation vs Radiologist ####################################

plt.figure(figsize=(10, 10))

sum_score = (
    df_final.score_tc_esq_sup + df_final.score_tc_esq_med + df_final.score_tc_esq_inf +
    df_final.score_tc_dir_sup + df_final.score_tc_dir_med + df_final.score_tc_dir_inf
)

# UNCOMMENT FOR DEBUG

# df_final["sum_score"] = sum_score
# df_final["one_minus_normal"] = 1 - df_final.seg_normal
# df_final = df_final.sort_values("sum_score")

# import pdb
# pdb.set_trace()

corr_coeff = (1 - df_final.seg_normal).corr(sum_score)
corr_coeff_str = ("%.2f" % corr_coeff).lstrip("0")
plt.scatter(sum_score, 1 - df_final.seg_normal, c="royalblue",
            label=f"Correlation coefficient = {corr_coeff_str}",
            s=df_final.volume_pulmao.apply(lambda x: (2 * x + 1) ** 2),
            alpha=0.7)

plt.xlabel("Ragiologist's score")
plt.ylabel("Affected lungs (%)")

props = dict(boxstyle="round", facecolor="snow", alpha=0.4)
textstr = f"Correlation coefficient = {corr_coeff_str}"
plt.text(0.05, 0.87, textstr, verticalalignment="center", bbox=props)

plt.grid(which="major")
plt.grid(which="minor", linestyle='--', alpha=0.4)
plt.minorticks_on()

save_path_corr = os.path.join(BASE_DIR, "exploratory", "rad_score_corr.tiff")
plt.savefig(save_path_corr, format="tiff", dpi=300)
plt.close()
