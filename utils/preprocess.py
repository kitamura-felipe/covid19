import numpy as np

from .helpers import estimate_p_healthy_lungs


def preprocess(df_final):
    # We account for known comorbidities. For this reason, the unknown cases, we zero this variable.

    df_final.n_comorbidades = df_final.n_comorbidades.fillna(0)

    # As we want to predict mortality, we can only take into account patients that either died
    # or were discharged.

    df_final = df_final[(df_final.alta) | (df_final.obito)]
    # df_final = df_final[df_final.instituicao.apply(str.upper) != "UNIFESP"]

    # Get the highest score for each lung

    df_final["max_dir_score"] = df_final.apply(
        lambda row: max(row["score_tc_dir_sup"], row["score_tc_dir_med"], row["score_tc_dir_inf"]),
        axis=1
    )

    df_final["max_esq_score"] = df_final.apply(
        lambda row: max(row["score_tc_esq_sup"], row["score_tc_esq_med"], row["score_tc_esq_inf"]),
        axis=1
    )

    df_final["sum_scores"] = df_final.apply(
        lambda row: sum([
            row["score_tc_dir_sup"], row["score_tc_dir_med"], row["score_tc_dir_inf"],
            row["score_tc_esq_sup"], row["score_tc_esq_med"], row["score_tc_esq_inf"]
        ]), axis=1
    )

    df_final["leucocitos"] = df_final["leucocitos"].apply(np.log1p)
    df_final["lactato"] = df_final["lactato"].apply(np.log1p)

    # We have two measures for the level of affectness of the lungs. One is provided by the physician,
    # the other is provided deep segmentation model. The former is more precise, but we have the first
    # for most of the TC scans. For this reason, we can use the score provided by the physician as an
    # alternative to fill the missing deep learning inferences.

    # Finally, we fill the missing values
    fill_seg_normal = df_final.apply(estimate_p_healthy_lungs, axis=1)
    # df_final.seg_normal = df_final.seg_normal.fillna(fill_seg_normal)  # Comment out this line if you do not want to fill NA

    ############################## Finished processing the data ##############################
    return df_final
