import funcy
import functools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from copy import deepcopy

font = {
    "size": 22
}
matplotlib.rc("font", **font)

def plot_results_2x2(summaries, save_fpath, fformat="pdf", dpi=300):
    fig = plt.figure(figsize=(20, 20))

    nrows, ncols = 2, 2

    ax1 = fig.add_subplot(nrows, ncols, 1)
    for summary in summaries:
        ax1.plot(summary["fpr"], summary["tpr"], lw=2, label="AUC = %0.2f" % summary["roc_auc"])
    ax1.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--")
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_ylabel("Sensitivity")
    ax1.set_xlabel("1 - Specificity")
    ax1.legend(loc="lower right")
    ax1.grid(which="major")
    ax1.grid(which="minor", linestyle='--', alpha=0.4)
    ax1.minorticks_on()

    ax2 = fig.add_subplot(nrows, ncols, 2)
    for summary in summaries:
        ax2.step(summary["recall"], summary["precision"], lw=2, label="Avg Prec = %0.2f" % summary["avg_precision"])
    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="lower left")
    ax2.grid(which="major")
    ax2.grid(which="minor", linestyle='--', alpha=0.4)
    ax2.minorticks_on()

    ax3 = fig.add_subplot(nrows, ncols, 3)
    ax3.plot([0., 1.], [0., 1.], "r--")
    for summary in summaries:
        ax3.plot(summary["mean_predicted_value"], summary["fraction_of_positives"], "s-")
    ax3.set_xlim([-0.05, 1.05])
    ax3.set_ylim([-0.05, 1.05])
    ax3.set_xlabel("Mean predicted value")
    ax3.set_ylabel("Fraction of positive cases")
    ax3.grid(which="major")
    ax3.grid(which="minor", linestyle='--', alpha=0.4)
    ax3.minorticks_on()

    df_summary = pd.DataFrame(functools.reduce(lambda l1, l2: l1+l2, [
        [
            ("ROC AUC", summary["roc_auc"]),
            ("Average\nPrecision", summary["avg_precision"]),
            ("Best F1", summary["best_f1"]),
            ("Best\nAccuracy", summary["best_acc"])
        ]
        for summary in summaries
    ]), columns=["Metric", "Value"])

    ax4 = fig.add_subplot(nrows, ncols, 4)
    sns.boxplot(x="Metric", y="Value", data=df_summary, width=0.3, palette="muted")
    ax4.set_ylim([0., 1.])
    ax4.grid(which="major")
    ax4.grid(which="minor", linestyle='--', alpha=0.4)
    ax4.minorticks_on()

    plt.tight_layout(pad=3)
    plt.savefig(save_fpath, format=fformat, dpi=dpi)
    plt.close()

    return df_summary.groupby("Metric").agg({
        "Value": ["mean", "std"]
    }).reset_index()


def plot_results_1x2(summaries, save_fpath, fformat="pdf", dpi=300):
    fig = plt.figure(figsize=(20, 10))

    nrows, ncols = 1, 2

    mean_fpr = np.linspace(0, 1, 100)
    aucs = []
    tprs = []
    for summary in summaries:
        aucs.append(summary["roc_auc"])

        interp_tpr = np.interp(mean_fpr, summary["fpr"], summary["tpr"])
        interp_tpr[0] = 0.0

        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc, std_auc = np.mean(aucs), np.std(aucs)

    ax1 = fig.add_subplot(nrows, ncols, 1)
    ax1.plot(
        mean_fpr, mean_tpr, color="b", lw=2, alpha=1.,
        label="Mean AUC = %0.2f $\pm$ %0.2f" % (mean_auc, std_auc)
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=.4, label=r"$\pm$ 1 std. dev.")

    ax1.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--")
    for summary in summaries:
        ax1.plot(summary["fpr"], summary["tpr"], "--", lw=2, label="AUC = %0.2f" % summary["roc_auc"], alpha=1.)

    ax1.legend(loc="lower right", framealpha=0.7)
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_ylabel("Sensitivity")
    ax1.set_xlabel("1 - Specificity")

    ax1.grid(which="major")
    ax1.grid(which="minor", linestyle='--', alpha=0.4)
    ax1.minorticks_on()

    df_summary = pd.DataFrame(functools.reduce(lambda l1, l2: l1+l2, [
        [
            ("ROC AUC", summary["roc_auc"]),
            ("Average\nPrecision", summary["avg_precision"]),
            ("Best F1", summary["best_f1"]),
            ("Best\nAccuracy", summary["best_acc"])
        ]
        for summary in summaries
    ]), columns=["Metric", "Value"])

    ax4 = fig.add_subplot(nrows, ncols, 2)
    sns.boxplot(x="Metric", y="Value", data=df_summary, width=0.3, palette="muted")
    ax4.set_ylim([0., 1.])
    ax4.grid(which="major")
    ax4.grid(which="minor", linestyle='--', alpha=0.4)
    ax4.minorticks_on()

    plt.tight_layout(pad=3)
    plt.savefig(save_fpath, format=fformat, dpi=dpi)
    plt.close()

    return df_summary.groupby("Metric").agg({
        "Value": ["mean", "std"]
    }).reset_index()


def plot_shap_values(X_test, summaries, feature_names, save_fpath, fformat="pdf", dpi=300):
    plt.figure()

    shap_values = np.stack([summary["shap_values"] for summary in summaries])
    shap_values = np.mean(shap_values, axis=0)

    shap.summary_plot(
        shap_values, X_test, plot_type="violin", plot_size=(10, 7), sort=False, show=False,
        feature_names=feature_names
    )
    plt.tight_layout()
    plt.savefig(save_fpath, format=fformat, dpi=dpi)
    plt.close()


def plot_survival(df_test, features, summaries, save_fpath, fformat="pdf", dpi=300):
    use_df = deepcopy(df_test)
    use_df["tempo_estadia_hospitalar"] = use_df.apply(
        lambda row: 100000 if not row["obito"] else row["tempo_estadia_hospitalar"],
        axis=1
    )

    grouped = use_df.groupby("record_id")
    data_tmp = []
    for record_id, group in grouped:
        sorted_group = group.sort_values("data")
        data_tmp.append(dict(sorted_group.iloc[0]))

    df_first_register = pd.DataFrame(data_tmp)
    X_test = df_first_register[features]

    basic_data = {
        "record_id": list(df_first_register.record_id),
        "tempo_estadia_hospitalar": list(df_first_register.tempo_estadia_hospitalar),
        "obito": list(df_first_register.obito)
    }

    plot_data = []
    survival_range = list(range(0, 51, 1))
    for i, summary in enumerate(summaries):
        model = summary["model"]
        best_thr_f1 = summary["best_thr_f1"]

        proba_preds = model.predict_proba(X_test)
        y_score = proba_preds[:, 1]
        target_pred = funcy.lmap(lambda v: int(v >= best_thr_f1), y_score)

        data_tmp = {"pred": target_pred}
        data_tmp.update(basic_data)
        df_survival = pd.DataFrame(data_tmp)

        pos = df_survival[df_survival.pred == 1]
        neg = df_survival[df_survival.pred == 0]

        pos_percent = [100 - 100 * len(pos[pos.tempo_estadia_hospitalar <= i]) / len(pos) for i in survival_range]
        neg_percent = [100 - 100 * len(neg[neg.tempo_estadia_hospitalar <= i]) / len(neg) for i in survival_range]

        plot_data.append(
            {
                "pos": pos_percent,
                "neg": neg_percent
            }
        )

    pos_data = np.array([d["pos"] for d in plot_data])
    pos_mean = np.mean(pos_data, axis=0)
    pos_std = np.std(pos_data, axis=0)
    pos_upper = pos_mean + pos_std
    pos_lower = pos_mean - pos_std

    neg_data = np.array([d["neg"] for d in plot_data])
    neg_mean = np.mean(neg_data, axis=0)
    neg_std = np.std(neg_data, axis=0)
    neg_upper = neg_mean + neg_std
    neg_lower = neg_mean - neg_std

    plt.figure(figsize=(15, 10))

    plt.plot(survival_range, pos_mean, "royalblue", label="Predicted positive")
    plt.fill_between(survival_range, pos_lower, pos_upper, color="royalblue", alpha=.4, label=r"$\pm$ 1 std. dev.")

    plt.plot(survival_range, neg_mean, "darkorange", label="Predicted negative")
    plt.fill_between(survival_range, neg_lower, neg_upper, color="darkorange", alpha=.4, label=r"$\pm$ 1 std. dev.")

    plt.legend([
        "Predicted positive",
        "Predicted negative"
    ], loc="lower left")

    plt.ylabel("Percentage of Survivors")
    plt.xlabel("Days")
    plt.ylim([0, 105])

    plt.grid(which="major")
    plt.grid(which="minor", linestyle='--', alpha=0.4)
    plt.minorticks_on()

    plt.savefig(save_fpath, format=fformat, dpi=dpi)
    plt.close()


def plot_sensitivity_specificity_vs_threshold(summaries, save_fpath, step=0.05, fformat="pdf", dpi=300):
    nrows, ncols = 1, 1
    fig = plt.figure(figsize=(15*ncols, 15*nrows))

    mean_thres = np.linspace(0, 1, 100)
    senss = []
    for summary in summaries:
        sens = np.array(summary["tpr"])
        thres = np.array(summary["thresholds"])
        thres[0] = 1.0

        interp_sens = np.interp(mean_thres, 1-thres, sens)
        senss.append(interp_sens)

    mean_sens = np.mean(senss, axis=0)
    std_sens = np.std(senss, axis=0)
    senss_upper = np.minimum(mean_sens + std_sens, 1)
    senss_lower = np.maximum(mean_sens - std_sens, 0)

    ax1 = fig.add_subplot(nrows, ncols, 1)
    for summary in summaries:
        sens = np.array(summary["tpr"])
        thres = np.array(summary["thresholds"])
        thres[0] = 1.0
        ax1.plot(thres, sens, "--", lw=2, alpha=1.)

    ax1.plot(1-mean_thres, mean_sens, color="b", lw=2, alpha=1., label="Mean sensitivity")
    ax1.fill_between(1-mean_thres, senss_lower, senss_upper, color="royalblue", alpha=.4, label=r"$\pm$ 1 std. dev.")

    mean_thres = np.linspace(0, 1, 100)
    specs = []
    for summary in summaries:
        fpr = np.array(summary["fpr"])
        thres = np.array(summary["thresholds"])
        thres[0] = 1.0

        interp_spec = np.interp(mean_thres, 1-thres, 1-fpr)
        specs.append(interp_spec)

    mean_spec = np.mean(specs, axis=0)
    std_spec = np.std(specs, axis=0)
    specs_upper = np.minimum(mean_spec + std_spec, 1)
    specs_lower = np.maximum(mean_spec - std_spec, 0)

    for summary in summaries:
        spec = 1 - np.array(summary["fpr"])
        thres = np.array(summary["thresholds"])
        thres[0] = 1.0
        ax1.plot(thres, spec, "--", lw=2, alpha=1.)

    ax1.plot(1-mean_thres, mean_spec, color="darkorange", lw=2, alpha=1., label="Mean specificity")
    ax1.fill_between(1-mean_thres, specs_lower, specs_upper, color="darkorange", alpha=.4, label=r"$\pm$ 1 std. dev.")

    ax1.legend(loc="center right", framealpha=0.7)
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_ylabel("Sensitivity / Specificity")
    ax1.set_xlabel("Cut-off value")

    ax1.grid(which="major")
    ax1.grid(which="minor", linestyle='--', alpha=0.4)
    ax1.minorticks_on()

    plt.tight_layout(pad=3)
    plt.savefig(save_fpath, format=fformat, dpi=dpi)
    plt.close()
