"""
Explainer — SHAP tabanlı açıklanabilirlik (XAI).

Klasik ML modellerinin tahminlerini açıklar:
  - Global: Summary plot (beeswarm), bar chart (mean |SHAP|)
  - Lokal: Waterfall plot (tek bir tahmin nasıl yapıldı)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import shap
from typing import Optional, Any
from dataclasses import dataclass

matplotlib.use("Agg")


@dataclass
class ShapResult:
    """SHAP hesaplama sonuçları."""
    shap_values: np.ndarray
    base_value: float
    feature_names: list
    X_sample: pd.DataFrame
    explainer_type: str


class Explainer:
    """SHAP ile model açıklanabilirliği.

    Otomatik explainer seçimi:
      - Tree-based modeller (XGBoost, LightGBM, RF, ExtraTrees, GBM) → TreeExplainer (hızlı)
      - Diğerleri (LR, SVM, AdaBoost) → KernelExplainer (yavaş, sample bazlı)
    """

    # Stil — Visualizer ile uyumlu dark theme
    COLORS = {
        "primary": "#e94560",
        "secondary": "#feca57",
        "accent": "#00d2ff",
        "success": "#7bed9f",
        "bg_dark": "#0f0c29",
        "bg_card": "#1a1a2e",
        "text": "#ffffff",
        "text_muted": "#a0a0b8",
    }

    TREE_MODEL_NAMES = (
        "XGBClassifier", "LGBMClassifier", "RandomForestClassifier",
        "ExtraTreesClassifier", "GradientBoostingClassifier",
    )

    def __init__(self, model: Any, X_background: pd.DataFrame, max_background: int = 100):
        """
        Args:
            model: Eğitilmiş sklearn-uyumlu model.
            X_background: Background dataset (KernelExplainer için referans).
            max_background: Background boyutu (KernelExplainer hız için sınırlanır).
        """
        self.model = model
        self.feature_names = list(X_background.columns)
        self._explainer_type = self._detect_explainer_type()

        if self._explainer_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        else:
            # Kernel explainer için küçük background sample
            bg = shap.sample(X_background, min(max_background, len(X_background)),
                             random_state=42)
            predict_fn = (model.predict_proba if hasattr(model, "predict_proba")
                          else model.predict)
            self.explainer = shap.KernelExplainer(predict_fn, bg)

    # ------------------------------------------------------------------
    def _detect_explainer_type(self) -> str:
        cls_name = type(self.model).__name__
        return "tree" if cls_name in self.TREE_MODEL_NAMES else "kernel"

    # ------------------------------------------------------------------
    def compute_shap_values(
        self,
        X: pd.DataFrame,
        max_samples: int = 200,
    ) -> ShapResult:
        """SHAP değerlerini hesapla.

        Args:
            X: Açıklanacak veriler.
            max_samples: KernelExplainer için maksimum örnek (hız).

        Returns:
            ShapResult.
        """
        if self._explainer_type == "kernel" and len(X) > max_samples:
            X_sample = X.sample(n=max_samples, random_state=42)
        else:
            X_sample = X

        if self._explainer_type == "tree":
            shap_values = self.explainer.shap_values(X_sample)
            # Bazı modeller (XGBoost binary) tek matris döndürür,
            # bazıları (RF binary) liste döndürür → pozitif sınıfı al
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif shap_values.ndim == 3:
                # (n_samples, n_features, n_classes) → pozitif sınıf
                shap_values = shap_values[:, :, 1]

            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(np.array(base_value).flatten()[-1])
        else:
            # KernelExplainer
            shap_values = self.explainer.shap_values(X_sample, silent=True)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(np.array(base_value).flatten()[-1])

        return ShapResult(
            shap_values=shap_values,
            base_value=float(base_value),
            feature_names=self.feature_names,
            X_sample=X_sample,
            explainer_type=self._explainer_type,
        )

    # ------------------------------------------------------------------
    # GLOBAL AÇIKLAMALAR
    # ------------------------------------------------------------------

    def plot_global_bar(self, result: ShapResult, top_n: int = 15):
        """Global önem — mean(|SHAP|) bar chart."""
        mean_abs = np.abs(result.shap_values).mean(axis=0)
        idx = np.argsort(mean_abs)[-top_n:]
        top_features = [result.feature_names[i] for i in idx]
        top_values = mean_abs[idx]

        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.4)))
        fig.patch.set_facecolor(self.COLORS["bg_dark"])
        ax.set_facecolor(self.COLORS["bg_card"])

        # Gradient renkler
        norm = top_values / top_values.max()
        colors = [plt.cm.magma(0.3 + 0.55 * v) for v in norm]

        bars = ax.barh(range(len(top_features)), top_values, color=colors,
                       edgecolor="white", linewidth=0.3, alpha=0.92)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=10, color=self.COLORS["text"])
        ax.set_xlabel("Ortalama |SHAP Değeri| — Tahmine Etki Büyüklüğü",
                      color=self.COLORS["text"], fontsize=11)
        ax.set_title(
            f"Global Özellik Önemi (SHAP) — Top {top_n}\n"
            "Modelin tahminlerinde en çok kullandığı özellikler",
            color=self.COLORS["secondary"], fontsize=13, weight="bold", pad=12,
        )

        # Bar üstü değerler
        for bar, val in zip(bars, top_values):
            ax.text(val + top_values.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8.5,
                    color=self.COLORS["text"], weight="bold")

        ax.tick_params(colors=self.COLORS["text"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(self.COLORS["text_muted"])
        ax.spines["bottom"].set_color(self.COLORS["text_muted"])
        ax.spines["left"].set_alpha(0.3)
        ax.spines["bottom"].set_alpha(0.3)
        ax.grid(True, axis="x", alpha=0.1)
        plt.tight_layout()
        return fig

    def plot_global_summary(self, result: ShapResult, top_n: int = 15):
        """Global SHAP summary (beeswarm) — özellik dağılım etkisi."""
        # SHAP'ın kendi summary plot'u — figure backend
        plt.figure(figsize=(10, max(5, top_n * 0.4)))
        shap.summary_plot(
            result.shap_values,
            result.X_sample,
            feature_names=result.feature_names,
            max_display=top_n,
            show=False,
            plot_size=None,
        )
        fig = plt.gcf()
        fig.patch.set_facecolor(self.COLORS["bg_dark"])

        # Tüm axes'i koyu temaya uydur
        for ax in fig.axes:
            ax.set_facecolor(self.COLORS["bg_card"])
            ax.tick_params(colors=self.COLORS["text"], labelsize=9)
            for spine in ax.spines.values():
                spine.set_color(self.COLORS["text_muted"])
                spine.set_alpha(0.3)
            ax.xaxis.label.set_color(self.COLORS["text"])
            ax.yaxis.label.set_color(self.COLORS["text"])
            ax.title.set_color(self.COLORS["secondary"])

        fig.suptitle(
            "SHAP Summary Plot — Özellik Değeri ve Tahmin Etkisi\n"
            "Sağ: anomali tahminini artırır  ·  Sol: azaltır  ·  Renk: özellik değeri",
            color=self.COLORS["secondary"], fontsize=12, weight="bold", y=1.02,
        )
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # LOKAL AÇIKLAMA
    # ------------------------------------------------------------------

    def plot_local_waterfall(
        self,
        result: ShapResult,
        sample_idx: int,
        top_n: int = 12,
    ):
        """Tek bir tahmin için waterfall plot — hangi özellik ne kadar etki etti."""
        shap_row = result.shap_values[sample_idx]
        x_row = result.X_sample.iloc[sample_idx]

        # Top N özelliği (mutlak etkiye göre)
        order = np.argsort(np.abs(shap_row))[-top_n:]
        feats = [result.feature_names[i] for i in order]
        shap_vals = shap_row[order]
        feat_vals = x_row.values[order]

        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.5)))
        fig.patch.set_facecolor(self.COLORS["bg_dark"])
        ax.set_facecolor(self.COLORS["bg_card"])

        colors = [self.COLORS["primary"] if v > 0 else self.COLORS["accent"]
                  for v in shap_vals]
        bars = ax.barh(range(len(feats)), shap_vals, color=colors,
                       edgecolor="white", linewidth=0.3, alpha=0.92)

        # Etiketler: özellik = değer
        ylabels = [f"{f}\n= {v:.3f}" if abs(v) < 1000 else f"{f}\n= {v:.0f}"
                   for f, v in zip(feats, feat_vals)]
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(ylabels, fontsize=9, color=self.COLORS["text"])

        # SHAP değer etiketleri
        for bar, val in zip(bars, shap_vals):
            x_pos = val + (0.005 if val >= 0 else -0.005)
            ha = "left" if val >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha=ha, fontsize=8.5,
                    color=self.COLORS["text"], weight="bold")

        ax.axvline(x=0, color=self.COLORS["text_muted"], linewidth=0.8, alpha=0.5)
        ax.set_xlabel("SHAP Değeri (anomali olasılığına etki)",
                      color=self.COLORS["text"], fontsize=11)

        # Tahmin özeti
        sum_shap = shap_row.sum()
        prediction_logit = result.base_value + sum_shap
        ax.set_title(
            f"Lokal Açıklama — Örnek #{sample_idx}\n"
            f"Baseline: {result.base_value:+.3f}  +  SHAP toplamı: {sum_shap:+.3f}  =  {prediction_logit:+.3f}",
            color=self.COLORS["secondary"], fontsize=12, weight="bold", pad=12,
        )

        # Lejant
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.COLORS["primary"], label="Anomali tahminini artırır"),
            Patch(facecolor=self.COLORS["accent"], label="Anomali tahminini azaltır"),
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc="lower right",
                  facecolor=self.COLORS["bg_card"], edgecolor="#333",
                  labelcolor=self.COLORS["text"])

        ax.tick_params(colors=self.COLORS["text"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(self.COLORS["text_muted"])
        ax.spines["bottom"].set_color(self.COLORS["text_muted"])
        ax.spines["left"].set_alpha(0.3)
        ax.spines["bottom"].set_alpha(0.3)
        ax.grid(True, axis="x", alpha=0.1)
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    def get_top_features(self, result: ShapResult, top_n: int = 5) -> list:
        """Global en önemli N özelliği döndür."""
        mean_abs = np.abs(result.shap_values).mean(axis=0)
        idx = np.argsort(mean_abs)[-top_n:][::-1]
        return [result.feature_names[i] for i in idx]
