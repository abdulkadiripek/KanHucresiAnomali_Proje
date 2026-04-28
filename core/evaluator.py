"""
Evaluator — Model değerlendirme, metrik hesaplama ve raporlama.
"""

import numpy as np
import pandas as pd
from typing import Optional, Any
from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
    log_loss,
)


@dataclass
class EvalResult:
    """Bir modelin değerlendirme sonuçları."""
    model_name: str
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    mcc: float
    kappa: float
    log_loss_val: Optional[float]
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    confusion_mat: np.ndarray
    classification_rep: str
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None


class Evaluator:
    """Model değerlendirme ve karşılaştırma.

    Sektörel metrikler:
      - Accuracy, Balanced Accuracy
      - Precision, Recall, F1-Score
      - Matthews Correlation Coefficient (MCC)
      - Cohen's Kappa
      - Log Loss
      - ROC-AUC, PR-AUC
      - Confusion Matrix
      - Detaylı classification report
    """

    def __init__(self):
        self._results: dict[str, EvalResult] = {}

    # ------------------------------------------------------------------
    def evaluate(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        cv_mean: Optional[float] = None,
        cv_std: Optional[float] = None,
    ) -> EvalResult:
        """Kapsamlı değerlendirme yap.

        Args:
            model_name: Model adı.
            y_true: Gerçek etiketler.
            y_pred: Tahmin edilen etiketler.
            y_proba: Olasılık tahminleri (ROC/PR curve için).
            cv_mean: Cross-validation ortalama skoru.
            cv_std: Cross-validation standart sapması.

        Returns:
            EvalResult dataclass.
        """
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        cr = classification_report(y_true, y_pred, target_names=["Normal", "Anomali"])

        # Olasılık tabanlı metrikler
        ll = None
        roc_auc_val = None
        pr_auc_val = None

        if y_proba is not None:
            try:
                ll = log_loss(y_true, y_proba)
            except Exception:
                ll = None

            try:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc_val = auc(fpr, tpr)
            except Exception:
                roc_auc_val = None

            try:
                pr_auc_val = average_precision_score(y_true, y_proba)
            except Exception:
                pr_auc_val = None

        result = EvalResult(
            model_name=model_name,
            accuracy=acc,
            balanced_accuracy=bal_acc,
            precision=prec,
            recall=rec,
            f1=f1,
            mcc=mcc,
            kappa=kappa,
            log_loss_val=ll,
            roc_auc=roc_auc_val,
            pr_auc=pr_auc_val,
            confusion_mat=cm,
            classification_rep=cr,
            cv_mean=cv_mean,
            cv_std=cv_std,
        )
        self._results[model_name] = result
        return result

    # ------------------------------------------------------------------
    def get_comparison_table(self) -> pd.DataFrame:
        """Tüm modellerin karşılaştırma tablosunu döndür."""
        rows = []
        for name, r in self._results.items():
            row = {
                "Model": name,
                "Accuracy": r.accuracy,
                "Bal. Acc.": r.balanced_accuracy,
                "Precision": r.precision,
                "Recall": r.recall,
                "F1-Score": r.f1,
                "MCC": r.mcc,
                "Kappa": r.kappa,
            }
            if r.roc_auc is not None:
                row["ROC-AUC"] = r.roc_auc
            if r.pr_auc is not None:
                row["PR-AUC"] = r.pr_auc
            if r.log_loss_val is not None:
                row["Log Loss"] = r.log_loss_val
            if r.cv_mean is not None:
                row["CV Mean (F1)"] = r.cv_mean
            if r.cv_std is not None:
                row["CV Std"] = r.cv_std
            rows.append(row)

        df = pd.DataFrame(rows).set_index("Model")
        return df.sort_values("F1-Score", ascending=False)

    # ------------------------------------------------------------------
    def get_best_model(self, metric: str = "F1-Score") -> str:
        """En iyi modelin adını döndür."""
        table = self.get_comparison_table()
        if metric not in table.columns:
            metric = "F1-Score"
        return table[metric].idxmax()

    # ------------------------------------------------------------------
    def get_roc_data(self, y_true, y_proba):
        """ROC curve verilerini döndür."""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc_val = auc(fpr, tpr)
        return fpr, tpr, roc_auc_val

    def get_pr_data(self, y_true, y_proba):
        """Precision-Recall curve verilerini döndür."""
        precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_proba)
        pr_auc_val = average_precision_score(y_true, y_proba)
        return precision_arr, recall_arr, pr_auc_val

    # ------------------------------------------------------------------
    @property
    def results(self) -> dict:
        return dict(self._results)

    def clear(self):
        """Tüm sonuçları temizle."""
        self._results.clear()
