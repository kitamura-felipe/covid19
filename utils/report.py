import pandas as pd

from scipy.stats import ks_2samp
from tabulate import tabulate


def generate_experiment_report(
    latex_tag, target, df_split, df_final, features, metrics_summary,
    train_valid_records, test_records, save_to=None
):
    train_idade_mean, train_idade_std = train_valid_records.idade.mean(), train_valid_records.idade.std()
    test_idade_mean, test_idade_std = test_records.idade.mean(), test_records.idade.std()

    _, idade_pvalue = ks_2samp(train_valid_records.idade, test_records.idade)

    train_sexo_counts = train_valid_records.sexo_M.value_counts()
    train_Pmasc = train_sexo_counts[1] / (train_sexo_counts[1] + train_sexo_counts[0])

    test_sexo_counts = test_records.sexo_M.value_counts()
    test_Pmasc = test_sexo_counts[1] / (test_sexo_counts[1] + test_sexo_counts[0])

    _, sexo_pvalue = ks_2samp(train_valid_records.sexo_M, test_records.sexo_M)

    roc_mean = "%.3f" % metrics_summary[metrics_summary.Metric == "ROC AUC"][("Value", "mean")].iloc[0]
    roc_std = "%.3f" % metrics_summary[metrics_summary.Metric == "ROC AUC"][("Value", "std")].iloc[0]

    f1_mean = "%.3f" % metrics_summary[metrics_summary.Metric == "Best F1"][("Value", "mean")].iloc[0]
    f1_std = "%.3f" % metrics_summary[metrics_summary.Metric == "Best F1"][("Value", "std")].iloc[0]

    acc_mean = "%.3f" % metrics_summary[metrics_summary.Metric == "Best\nAccuracy"][("Value", "mean")].iloc[0]
    acc_std = "%.3f" % metrics_summary[metrics_summary.Metric == "Best\nAccuracy"][("Value", "std")].iloc[0]

    prec_mean = "%.3f" % metrics_summary[metrics_summary.Metric == "Average\nPrecision"][("Value", "mean")].iloc[0]
    prec_std = "%.3f" % metrics_summary[metrics_summary.Metric == "Average\nPrecision"][("Value", "std")].iloc[0]

    missing_values = tabulate(
        pd.DataFrame([
            (k, v, '%0.2f' % (100 * v / len(df_final))) for k, v in
            sorted(df_final.isna().sum().items(), key=lambda t: t[1], reverse=True)
            if k in features
        ], columns=["Feature", "Number of null values", "% of null values"]),
        headers="keys", tablefmt="psql"
    )

    metrics = tabulate(metrics_summary, headers="keys", tablefmt="psql")

    metrics_latex = f"""
{latex_tag} & ${roc_mean} \pm {roc_std}$ & ${prec_mean} \pm {prec_std}$ & ${f1_mean} \pm {f1_std}$ & ${acc_mean} \pm {acc_std}$ \\\\
"""

    stats_report = f"""
Target present = {df_split[df_split[target] == 1.0].record_id.nunique()}
Target absent = {df_split[df_split[target] == 0.0].record_id.nunique()}

Train/validation patients = {len(train_valid_records)}
Test patients = {len(test_records)}

Idade:

Train/Validation - {train_idade_mean} +- {train_idade_std}
Test - {test_idade_mean} +- {test_idade_std}

p value - {idade_pvalue}

Sexo:

Train/Validation - {100 * train_Pmasc}% Homens ({train_sexo_counts[1]}/{train_sexo_counts[1] + train_sexo_counts[0]})
Test - {100 * test_Pmasc}% Homens ({test_sexo_counts[1]}/{test_sexo_counts[1] + test_sexo_counts[0]})

p value - {sexo_pvalue}
"""

    train_Pmasc_str = "%.1f" % (100 * train_Pmasc)
    test_Pmasc_str = "%.1f" % (100 * test_Pmasc)

    stats_latex = f"""
\multirow{{3}}{{*}}{{{latex_tag}}} & Train/Validation & ${"%.2f" % train_idade_mean} \pm {"%.2f" % train_idade_std}$ &  {train_Pmasc_str}\% ({train_sexo_counts[1]}/{train_sexo_counts[1] + train_sexo_counts[0]}) \\\\ \\cline{{2-4}}
& Test & ${"%.2f" % test_idade_mean} \pm {"%.2f" % test_idade_std}$ & {test_Pmasc_str}\% ({test_sexo_counts[1]}/{test_sexo_counts[1] + test_sexo_counts[0]}) \\\\ \\cline{{2-4}}
& p-value & {"%.4f" % idade_pvalue} & {"%.4f" % sexo_pvalue} \\\\
\midrule
"""

    if save_to is not None:
        with open(save_to, "w") as f:
            f.write(stats_report)
            f.write("\n")
            f.write(missing_values)
            f.write("\n")
            f.write(metrics)
            f.write("\n")
            f.write(metrics_latex)
            f.write("\n")
            f.write(stats_latex)
            f.write("\n")

    return {
        "missing_values": missing_values,
        "metrics": metrics,
        "stats_report": stats_report,
    }
