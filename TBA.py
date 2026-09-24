

import math
import numpy as np


class TBA:

    def __init__(self, t=0.0, prevalence=None, labels=None, smoothing=0.0):
        self.t = float(t)
        self.prevalence = None if prevalence is None else np.asarray(prevalence, float)
        self.labels = None if labels is None else np.asarray(labels)
        self.smoothing = float(smoothing)
        if self.prevalence is not None:
            if np.any(self.prevalence <= 0):
                raise ValueError("all prevalences must be > 0")
            self.prevalence = self.prevalence / self.prevalence.sum()

    # ------------------------------------------------------------------ core
    @staticmethod
    def weights(p, t):
        """w_k(t) = p_k^t / sum_j p_j^t."""
        p = np.asarray(p, float)
        w = p ** t
        return w / w.sum()

    @staticmethod
    def recalls(cm, smoothing=0.0):
        """Per-class recall from a confusion matrix (rows = true, cols = pred)."""
        cm = np.asarray(cm, float)
        tp = np.diag(cm)
        n = cm.sum(axis=1)
        if smoothing > 0:
            return (tp + smoothing) / (n + 2 * smoothing)
        with np.errstate(invalid="ignore", divide="ignore"):
            return tp / n

    @staticmethod
    def compute(p, r, t):
        """tBA_t from prevalence vector p and recall vector r."""
        r = np.asarray(r, float)
        if np.any(np.isnan(r)):
            raise ValueError("recall is undefined for a class with zero support; "
                             "use smoothing > 0 or pass prevalence and drop the class")
        return float(TBA.weights(p, t) @ r)

    # --------------------------------------------------------------- scoring
    def confusion_matrix(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        labels = self.labels
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))
        idx = {lab: i for i, lab in enumerate(labels)}
        K = len(labels)
        cm = np.zeros((K, K), dtype=int)
        for yt, yp in zip(y_true, y_pred):
            cm[idx[yt], idx[yp]] += 1
        return cm, labels

    def from_confusion_matrix(self, cm, t=None):
        """tBA from a confusion matrix (rows = true, cols = predicted)."""
        t = self.t if t is None else t
        cm = np.asarray(cm, float)
        p = self.prevalence
        if p is None:
            p = cm.sum(axis=1) / cm.sum()
        return self.compute(p, self.recalls(cm, self.smoothing), t)

    def score(self, y_true, y_pred, t=None):
        """tBA from label vectors (same signature as sklearn scorers)."""
        cm, _ = self.confusion_matrix(y_true, y_pred)
        return self.from_confusion_matrix(cm, t)

    def __call__(self, y_true, y_pred):
        return self.score(y_true, y_pred)

    def curve(self, y_true=None, y_pred=None, cm=None, ts=None):
        """tBA_t over a grid of t; returns (ts, values)."""
        if cm is None:
            cm, _ = self.confusion_matrix(y_true, y_pred)
        ts = np.linspace(-1, 1, 41) if ts is None else np.asarray(ts, float)
        return ts, np.array([self.from_confusion_matrix(cm, t) for t in ts])

    # -------------------------------------------------------------- helpers
    def accuracy(self, y_true, y_pred):
        return self.score(y_true, y_pred, t=1.0)

    def balanced_accuracy(self, y_true, y_pred):
        return self.score(y_true, y_pred, t=0.0)

    @staticmethod
    def majority_predictor_score(p, t):
        """Closed-form tBA_t of the always-majority classifier: w_max(t)."""
        p = np.asarray(p, float)
        return float(TBA.weights(p, t)[np.argmax(p)])

    @staticmethod
    def chance_level(K):
        """tBA_t of a uniform random classifier, for every t."""
        return 1.0 / K

    @staticmethod
    def adjusted(value, K):
        """Chance-corrected tBA: (tBA - 1/K) / (1 - 1/K)."""
        return (value - 1.0 / K) / (1.0 - 1.0 / K)

    @staticmethod
    def t_from_cost_ratio(c, p_k, p_j):
        """t such that class k (prevalence p_k) counts c times class j."""
        return math.log(c) / math.log(p_k / p_j)

    @staticmethod
    def crossover_t(p, r1, r2, lo=-3.0, hi=3.0, tol=1e-10):
        """t* at which two recall vectors give equal tBA_t; None if no crossing."""
        f = lambda t: TBA.compute(p, r1, t) - TBA.compute(p, r2, t)
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0:
            return None
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if flo * fm <= 0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        return 0.5 * (lo + hi)

    def __repr__(self):
        return f"TBA(t={self.t}, smoothing={self.smoothing})"


if __name__ == "__main__":
    # sanity checks against the paper's hand-calculated cases
    m = TBA()
    cm1 = [[93, 2], [2, 3]]
    print("Case 1 : Acc", round(m.from_confusion_matrix(cm1, 1), 4),
          "tBA_0.5", round(m.from_confusion_matrix(cm1, 0.5), 4),
          "BA", round(m.from_confusion_matrix(cm1, 0), 4),
          "tBA_-0.5", round(m.from_confusion_matrix(cm1, -0.5), 4))
    cm5 = [[76, 3, 1], [4, 10, 1], [2, 2, 1]]
    print("Case 5 :", [round(m.from_confusion_matrix(cm5, t), 4) for t in (1, .5, 0, -.5, -1)])
    print("Case 8 crossover t* =", round(TBA.crossover_t([0.9, 0.1], [88/90, 0.4], [80/90, 0.9]), 4))
    print("Case 9 t =", round(TBA.t_from_cost_ratio(3, 0.05, 0.80), 4))
    print("Majority 95:5 at t=-0.5 =", round(TBA.majority_predictor_score([0.95, 0.05], -0.5), 4))
    # label-vector use with full-dataset prevalence and smoothing
    y_true = [0]*95 + [1]*5
    y_pred = [0]*93 + [1]*2 + [0]*2 + [1]*3
    print("score()  t=0.5 =", round(TBA(t=0.5).score(y_true, y_pred), 4))
    print("absent class, smoothing=1:",
          round(TBA(t=0, prevalence=[0.8, 0.15, 0.05], labels=[0, 1, 2], smoothing=1)
                .score([0]*80 + [1]*15, [0]*76 + [1]*3 + [2]*1 + [0]*4 + [1]*10 + [2]*1), 4))
