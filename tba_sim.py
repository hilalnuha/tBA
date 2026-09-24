
import argparse
import math
import os
import warnings
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
 
warnings.filterwarnings("ignore")
 
OUT = "tba_out"
os.makedirs(OUT, exist_ok=True)
 
# fixed categorical palette (assigned in order, never cycled)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.color": "#dddddd",
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.8, "figure.dpi": 150, "savefig.bbox": "tight",
})
 
T_GRID = np.linspace(-1.0, 1.0, 41)
T_REPORT = [1.0, 0.5, 0.0, -0.5, -1.0]
 
 
# --------------------------------------------------------------------------- #
# 1. Metric definitions
# --------------------------------------------------------------------------- #
def per_class_recall(cm, a=0.0):
    """Per-class recall from a confusion matrix (rows = true, cols = predicted).
    a > 0 applies Laplace smoothing (TP + a) / (n_k + 2a) so an absent class
    gets 0.5 instead of NaN."""
    cm = np.asarray(cm, dtype=float)
    n_k = cm.sum(axis=1)
    tp = np.diag(cm)
    if a > 0:
        return (tp + a) / (n_k + 2 * a)
    with np.errstate(invalid="ignore", divide="ignore"):
        return tp / n_k
 
 
def tba_weights(p, t):
    p = np.asarray(p, dtype=float)
    w = p ** t
    return w / w.sum()
 
 
def tba(p, r, t):
    """Tempered balanced accuracy from prevalence vector p and recall vector r."""
    return float(tba_weights(p, t) @ np.asarray(r, dtype=float))
 
 
def tba_from_cm(cm, t, p=None, a=0.0):
    cm = np.asarray(cm, dtype=float)
    if p is None:
        p = cm.sum(axis=1) / cm.sum()
    return tba(p, per_class_recall(cm, a), t)
 
 
def tba_score(y_true, y_pred, t, p=None, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return tba_from_cm(cm, t, p=p)
 
 
def tba_majority(p, t):
    """Closed-form score of the majority-class predictor (Property 2)."""
    return float(tba_weights(p, t)[np.argmax(p)])
 
 
def t_from_cost_ratio(c, p_k, p_j):
    """Property 3: t such that class k counts c times class j."""
    return math.log(c) / math.log(p_k / p_j)
 
 
def gmean(r):
    return float(np.prod(r) ** (1.0 / len(r)))
 
 
def crossover_t(p, r1, r2, lo=-3.0, hi=3.0):
    """t* where tBA_t(r1) == tBA_t(r2); None if no sign change on [lo, hi]."""
    f = lambda t: tba(p, r1, t) - tba(p, r2, t)
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
 
 
# --------------------------------------------------------------------------- #
# 2. Verification of the hand-calculated cases
# --------------------------------------------------------------------------- #
CASES = {
    "1  95:5 ordinary":        (np.array([[93, 2], [2, 3]]), None),
    "2  95:5 majority":        (np.array([[95, 0], [5, 0]]), None),
    "3  95:5 over-prediction": (np.array([[70, 25], [0, 5]]), None),
    "4a 999:1 detected":       (np.array([[990, 9], [0, 1]]), None),
    "4b 999:1 missed":         (np.array([[990, 9], [1, 0]]), None),
    "5  80:15:5":              (np.array([[76, 3, 1], [4, 10, 1], [2, 2, 1]]), None),
    "6a 5-class collapse":     (None, ([0.5, 0.2, 0.15, 0.1, 0.05], [0.9, 0.85, 0.8, 0.7, 0.0])),
    "6b 5-class uniform":      (None, ([0.5, 0.2, 0.15, 0.1, 0.05], [0.8, 0.75, 0.7, 0.65, 0.6])),
    "7  85:8:5:2 intrusion":   (None, ([0.85, 0.08, 0.05, 0.02], [0.98, 0.9, 0.8, 0.5])),
    "8-1 90:10 conservative":  (np.array([[88, 2], [6, 4]]), None),
    "8-2 90:10 aggressive":    (np.array([[80, 10], [1, 9]]), None),
}
 
 
def verify_cases():
    rows = []
    curves = {}
    for name, (cm, pr) in CASES.items():
        if cm is not None:
            p = cm.sum(axis=1) / cm.sum()
            r = per_class_recall(cm)
        else:
            p, r = map(np.array, pr)
        row = {"case": name, "p": ":".join(f"{x:g}" for x in p),
               "r": ", ".join(f"{x:.4f}" for x in r)}
        for t in T_REPORT:
            row[f"t={t:g}"] = round(tba(p, r, t), 4)
        row["G-mean"] = round(gmean(r), 4)
        rows.append(row)
        curves[name] = [tba(p, r, t) for t in T_GRID]
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/cases_verification.csv", index=False)
    print("\n=== Hand-calculated cases (t=1 is Acc, t=0 is BA) ===")
    print(df.to_string(index=False))
 
    # Case 8 crossover and Case 9 cost-ratio t
    p8 = np.array([0.9, 0.1])
    ts = crossover_t(p8, [88 / 90, 0.4], [80 / 90, 0.9])
    print(f"\nCase 8 crossover t* = {ts:.4f}")
    t9 = t_from_cost_ratio(3, 0.05, 0.80)
    w9 = tba_weights([0.8, 0.15, 0.05], t9)
    print(f"Case 9: t = {t9:.4f}, w = {np.round(w9, 4)}, w3/w1 = {w9[2] / w9[0]:.4f}, "
          f"tBA = {tba([0.8, 0.15, 0.05], [0.95, 10 / 15, 0.2], t9):.4f}")
    print(f"Property 3 checks: c=5 @95:5 -> t={t_from_cost_ratio(5, .05, .95):.4f}; "
          f"c=10 @999:1 -> t={t_from_cost_ratio(10, .001, .999):.4f}")
 
    # Figure: tBA vs t for selected cases
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    sel = ["1  95:5 ordinary", "2  95:5 majority", "3  95:5 over-prediction",
           "5  80:15:5", "8-1 90:10 conservative", "8-2 90:10 aggressive"]
    for i, name in enumerate(sel):
        ax.plot(T_GRID, curves[name], color=PALETTE[i], label=name.strip())
    ax.axvline(0, color="#555555", ls="--", lw=0.8)
    ax.text(0.02, 0.03, "BA", transform=ax.get_xaxis_transform(), fontsize=8)
    ax.text(0.93, 0.03, "Acc", transform=ax.get_xaxis_transform(), fontsize=8)
    ax.set_xlabel("tempering exponent $t$"); ax.set_ylabel("tBA$_t$")
    ax.set_xlim(-1, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_curves_cases.{ext}")
    plt.close(fig)
 
    # Figure: majority-class predictor closed form for several imbalance ratios
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    tt = np.linspace(-2, 1, 121)
    for i, p1 in enumerate([0.6, 0.8, 0.95, 0.99, 0.999]):
        p = np.array([p1, 1 - p1])
        ax.plot(tt, [tba_majority(p, t) for t in tt], color=PALETTE[i],
                label=f"{p1:g}:{1 - p1:g}")
    ax.axhline(0.5, color="#555555", ls=":", lw=0.8)
    ax.axvline(0, color="#555555", ls="--", lw=0.8)
    ax.set_xlabel("tempering exponent $t$"); ax.set_ylabel("tBA$_t$ of majority predictor")
    ax.set_xlim(-2, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False, title="majority : minority", title_fontsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_majority_curve.{ext}")
    plt.close(fig)
    return df
 
 
# --------------------------------------------------------------------------- #
# 3. Synthetic experiment
# --------------------------------------------------------------------------- #
def make_models(seed):
    return {
        "LogReg": LogisticRegression(max_iter=2000, random_state=seed),
        "LogReg-balanced": LogisticRegression(max_iter=2000, class_weight="balanced",
                                              random_state=seed),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "RF-balanced": RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                              random_state=seed),
        "kNN": KNeighborsClassifier(n_neighbors=5),
    }
 
 
SCENARIOS = {
    # name: (n_classes, class weights)
    "binary 90:10":  (2, [0.90, 0.10]),
    "binary 99:1":   (2, [0.99, 0.01]),
    "3-class 80:15:5": (3, [0.80, 0.15, 0.05]),
    "4-class 85:8:5:2": (4, [0.85, 0.08, 0.05, 0.02]),
    "5-class 50:20:15:10:5": (5, [0.50, 0.20, 0.15, 0.10, 0.05]),
}
 
 
def synthetic_experiment(n_rep=5, n_samples=6000):
    rows, curve_rows = [], []
    for sname, (K, weights) in SCENARIOS.items():
        for rep in range(n_rep):
            seed = 1000 + rep
            X, y = make_classification(
                n_samples=n_samples, n_features=20, n_informative=8, n_redundant=4,
                n_classes=K, n_clusters_per_class=1, weights=weights,
                class_sep=0.9, flip_y=0.02, random_state=seed)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=seed)
            p_full = np.bincount(y, minlength=K) / len(y)   # prevalence from full data
            labels = list(range(K))
            for mname, model in make_models(seed).items():
                model.fit(Xtr, ytr)
                yp = model.predict(Xte)
                cm = confusion_matrix(yte, yp, labels=labels)
                r = per_class_recall(cm)
                row = {"scenario": sname, "K": K, "rep": rep, "model": mname,
                       "Acc": tba(p_full, r, 1.0),
                       "BA": balanced_accuracy_score(yte, yp),
                       "G-mean": gmean(r),
                       "MCC": matthews_corrcoef(yte, yp),
                       "macro-F1": f1_score(yte, yp, average="macro"),
                       "min recall": r.min()}
                for t in T_REPORT:
                    row[f"tBA t={t:g}"] = tba(p_full, r, t)
                rows.append(row)
                for t in T_GRID:
                    curve_rows.append({"scenario": sname, "rep": rep, "model": mname,
                                       "t": t, "tBA": tba(p_full, r, t)})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/synthetic_results.csv", index=False)
    summary = df.groupby(["scenario", "model"]).mean(numeric_only=True).drop(
        columns=["K", "rep"]).round(4)
    summary.to_csv(f"{OUT}/synthetic_summary.csv")
    print("\n=== Synthetic experiment: mean over repetitions ===")
    pd.set_option("display.width", 200)
    print(summary.to_string())
 
    # Figure: mean tBA vs t per model, one panel per scenario
    cdf = pd.DataFrame(curve_rows)
    scen = list(SCENARIOS)
    fig, axes = plt.subplots(1, len(scen), figsize=(2.3 * len(scen), 2.6), sharey=True)
    models = list(make_models(0))
    for ax, sname in zip(axes, scen):
        sub = cdf[cdf.scenario == sname].groupby(["model", "t"]).tBA.mean().unstack(0)
        for i, m in enumerate(models):
            ax.plot(sub.index, sub[m], color=PALETTE[i], label=m)
        ax.axvline(0, color="#555555", ls="--", lw=0.8)
        ax.set_title(sname, fontsize=8); ax.set_xlabel("$t$"); ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("tBA$_t$")
    axes[0].legend(fontsize=6.5, frameon=False, loc="lower right")
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_synthetic_curves.{ext}")
    plt.close(fig)
    return df
 
 
# --------------------------------------------------------------------------- #
# 4. Ranking disagreement between Acc and BA, and the crossover t*
# --------------------------------------------------------------------------- #
def ranking_analysis(df):
    rows, tstars = [], []
    for (sname, rep), g in df.groupby(["scenario", "rep"]):
        K = int(g.K.iloc[0])
        p = np.array(SCENARIOS[sname][1])
        models = g.set_index("model")
        names = list(models.index)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = models.loc[names[i]], models.loc[names[j]]
                acc_pref = np.sign(a["Acc"] - b["Acc"])
                ba_pref = np.sign(a["BA"] - b["BA"])
                disagree = acc_pref * ba_pref < 0
                rows.append({"scenario": sname, "rep": rep, "model A": names[i],
                             "model B": names[j], "Acc prefers": names[i] if acc_pref > 0 else names[j],
                             "BA prefers": names[i] if ba_pref > 0 else names[j],
                             "disagree": bool(disagree)})
    rdf = pd.DataFrame(rows)
    rdf.to_csv(f"{OUT}/ranking_disagreement.csv", index=False)
    rate = rdf.groupby("scenario").disagree.mean()
    print("\n=== Fraction of classifier pairs on which Acc and BA disagree ===")
    print(rate.round(3).to_string())
 
    # crossover t* for the disagreeing pairs, recomputed from recalls
    # (recalls are recovered by re-fitting is expensive, so use the stored tBA grid instead)
    for (sname, rep), g in df.groupby(["scenario", "rep"]):
        p = np.array(SCENARIOS[sname][1])
        # reconstruct recall vectors is not stored; approximate crossover from
        # the five reported t values by linear interpolation of the difference
        models = g.set_index("model")
        names = list(models.index)
        cols = [f"tBA t={t:g}" for t in sorted(T_REPORT)]
        tt = sorted(T_REPORT)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                d = (models.loc[names[i], cols] - models.loc[names[j], cols]).values.astype(float)
                for k in range(len(tt) - 1):
                    if d[k] * d[k + 1] < 0:
                        tstars.append(tt[k] - d[k] * (tt[k + 1] - tt[k]) / (d[k + 1] - d[k]))
    if tstars:
        fig, ax = plt.subplots(figsize=(3.6, 2.5))
        ax.hist(tstars, bins=np.linspace(-1, 1, 21), color=PALETTE[0],
                edgecolor="white", linewidth=0.8)
        ax.axvline(0, color="#555555", ls="--", lw=0.8)
        ax.set_xlabel("crossover $t^*$ between two classifiers")
        ax.set_ylabel("count")
        for ext in ("pdf", "png"):
            fig.savefig(f"{OUT}/fig_crossover_hist.{ext}")
        plt.close(fig)
        print(f"\n{len(tstars)} ranking crossovers found on [-1,1]; "
              f"median t* = {np.median(tstars):.3f}, "
              f"{np.mean(np.array(tstars) > 0):.0%} lie in (0,1) i.e. between BA and Acc")
    return rdf
 
 
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_rep = 2 if args.quick else 5
 
    verify_cases()
    df = synthetic_experiment(n_rep=n_rep)
    ranking_analysis(df)
    print(f"\nAll outputs written to ./{OUT}/")
 
 
if __name__ == "__main__":
    main()
