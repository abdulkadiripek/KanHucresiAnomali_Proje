"""
Visualizer — Tüm grafik ve görselleştirme işlemleri.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import Optional, Any

matplotlib.use("Agg")


class Visualizer:
    """Profesyonel görselleştirme sınıfı.

    Tüm grafikler dark theme ile uyumlu, yüksek kaliteli çıktı üretir.
    """

    # Renk paleti
    COLORS = {
        "primary": "#e94560",
        "secondary": "#feca57",
        "accent": "#00d2ff",
        "success": "#7bed9f",
        "bg_dark": "#0f0c29",
        "bg_card": "#1a1a2e",
        "text": "#ffffff",
        "text_muted": "#a0a0b8",
        "gradient": ["#e94560", "#feca57", "#00d2ff", "#7bed9f", "#a29bfe", "#fd79a8"],
    }

    def __init__(self, figsize: tuple = (10, 6)):
        self.figsize = figsize

    # ------------------------------------------------------------------
    # Temel Stil
    # ------------------------------------------------------------------

    def _apply_style(self, fig, ax):
        """Dark theme stilini uygula."""
        fig.patch.set_facecolor(self.COLORS["bg_dark"])
        ax.set_facecolor(self.COLORS["bg_card"])
        ax.tick_params(colors=self.COLORS["text"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(self.COLORS["text_muted"])
        ax.spines["bottom"].set_color(self.COLORS["text_muted"])
        ax.spines["left"].set_alpha(0.3)
        ax.spines["bottom"].set_alpha(0.3)

    # ------------------------------------------------------------------
    # Grafikler
    # ------------------------------------------------------------------

    def plot_confusion_matrix(self, y_true, y_pred, model_name: str, f1: float):
        """Confusion Matrix heatmap."""
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        self._apply_style(fig, ax)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="magma",
            xticklabels=["Normal (0)", "Anomali (1)"],
            yticklabels=["Normal (0)", "Anomali (1)"],
            linewidths=2, linecolor=self.COLORS["bg_dark"],
            annot_kws={"size": 22, "weight": "bold", "color": "white"},
            ax=ax, cbar_kws={"shrink": 0.8},
        )
        ax.set_xlabel("Tahmin Edilen", fontsize=13, color=self.COLORS["text"], labelpad=10)
        ax.set_ylabel("Gerçek", fontsize=13, color=self.COLORS["text"], labelpad=10)
        ax.set_title(
            f"Confusion Matrix — {model_name}\nF1-Score: {f1:.4f}",
            fontsize=14, weight="bold", color=self.COLORS["secondary"], pad=15,
        )
        plt.tight_layout()
        return fig

    def plot_roc_curves(self, roc_data: dict):
        """Birden fazla modelin ROC Curve karşılaştırması.

        Args:
            roc_data: {model_name: (fpr, tpr, auc_score)} dict.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        self._apply_style(fig, ax)

        for i, (name, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
            color = self.COLORS["gradient"][i % len(self.COLORS["gradient"])]
            ax.plot(fpr, tpr, color=color, lw=2.5,
                    label=f"{name} (AUC = {auc_val:.4f})")

        ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.3)
        ax.set_xlabel("False Positive Rate", color=self.COLORS["text"], fontsize=12)
        ax.set_ylabel("True Positive Rate", color=self.COLORS["text"], fontsize=12)
        ax.set_title("ROC Curve Karşılaştırması", color=self.COLORS["secondary"],
                      fontsize=14, weight="bold", pad=15)
        ax.legend(loc="lower right", fontsize=9, facecolor=self.COLORS["bg_card"],
                  edgecolor="#333", labelcolor=self.COLORS["text"])
        ax.grid(True, alpha=0.1)
        plt.tight_layout()
        return fig

    def plot_pr_curves(self, pr_data: dict):
        """Precision-Recall Curve karşılaştırması.

        Args:
            pr_data: {model_name: (precision, recall, ap)} dict.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        self._apply_style(fig, ax)

        for i, (name, (prec, rec, ap)) in enumerate(pr_data.items()):
            color = self.COLORS["gradient"][i % len(self.COLORS["gradient"])]
            ax.plot(rec, prec, color=color, lw=2.5,
                    label=f"{name} (AP = {ap:.4f})")

        ax.set_xlabel("Recall", color=self.COLORS["text"], fontsize=12)
        ax.set_ylabel("Precision", color=self.COLORS["text"], fontsize=12)
        ax.set_title("Precision-Recall Curve Karşılaştırması",
                      color=self.COLORS["secondary"], fontsize=14, weight="bold", pad=15)
        ax.legend(loc="lower left", fontsize=9, facecolor=self.COLORS["bg_card"],
                  edgecolor="#333", labelcolor=self.COLORS["text"])
        ax.grid(True, alpha=0.1)
        plt.tight_layout()
        return fig

    def plot_feature_importance(self, model, feature_names: list,
                                 model_name: str, top_n: int = 15):
        """Feature importance yatay bar chart."""
        if not hasattr(model, "feature_importances_"):
            return None

        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(9, 6))
        self._apply_style(fig, ax)

        # Gradient bar colors
        norm_vals = top_importances / top_importances.max()
        colors = [plt.cm.magma(0.3 + 0.6 * v) for v in norm_vals]

        bars = ax.barh(range(len(top_features)), top_importances,
                       color=colors, edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=9, color=self.COLORS["text"])
        ax.set_xlabel("Önem Skoru", color=self.COLORS["text"], fontsize=12)
        ax.set_title(f"Top {top_n} Özellik — {model_name}",
                      color=self.COLORS["secondary"], fontsize=14, weight="bold", pad=15)
        ax.grid(True, axis="x", alpha=0.1)
        plt.tight_layout()
        return fig

    def plot_metric_comparison(self, comparison_df: pd.DataFrame):
        """Model karşılaştırma grouped bar chart."""
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        available = [m for m in metrics if m in comparison_df.columns]
        if not available:
            return None

        fig, ax = plt.subplots(figsize=(11, 5.5))
        self._apply_style(fig, ax)

        x = np.arange(len(comparison_df.index))
        width = 0.17
        colors = [self.COLORS["primary"], self.COLORS["secondary"],
                  self.COLORS["accent"], self.COLORS["success"]]

        for i, metric in enumerate(available):
            bars = ax.bar(x + i * width, comparison_df[metric], width,
                          label=metric, color=colors[i % len(colors)],
                          edgecolor="white", linewidth=0.2, alpha=0.92)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7,
                        color=self.COLORS["text"], weight="bold")

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(comparison_df.index, fontsize=11, color=self.COLORS["text"])
        ax.set_ylabel("Skor", color=self.COLORS["text"], fontsize=12)
        ax.set_title("Model Performans Karşılaştırması", color=self.COLORS["secondary"],
                      fontsize=14, weight="bold", pad=15)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=9, facecolor=self.COLORS["bg_card"],
                  edgecolor="#333", labelcolor=self.COLORS["text"], loc="upper left")
        ax.grid(True, axis="y", alpha=0.1)
        plt.tight_layout()
        return fig

    def plot_cv_scores(self, cv_data: dict):
        """Cross-validation skor dağılımı box/violin plot.

        Args:
            cv_data: {model_name: cv_scores_array} dict.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        self._apply_style(fig, ax)

        names = list(cv_data.keys())
        scores = list(cv_data.values())

        parts = ax.violinplot(scores, positions=range(len(names)),
                               showmeans=True, showmedians=True)

        for i, pc in enumerate(parts.get("bodies", [])):
            color = self.COLORS["gradient"][i % len(self.COLORS["gradient"])]
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        if "cmeans" in parts:
            parts["cmeans"].set_color(self.COLORS["secondary"])
        if "cmedians" in parts:
            parts["cmedians"].set_color(self.COLORS["text"])
        for key in ["cbars", "cmins", "cmaxes"]:
            if key in parts:
                parts[key].set_color(self.COLORS["text_muted"])

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=10, color=self.COLORS["text"])
        ax.set_ylabel("F1-Score", color=self.COLORS["text"], fontsize=12)
        ax.set_title("Cross-Validation Skor Dağılımı", color=self.COLORS["secondary"],
                      fontsize=14, weight="bold", pad=15)
        ax.grid(True, axis="y", alpha=0.1)
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # EDA Grafikleri
    # ------------------------------------------------------------------

    def plot_class_distribution(self, labels: pd.Series):
        """Anomali sınıf dağılımı pie chart."""
        counts = labels.value_counts()
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor(self.COLORS["bg_dark"])

        colors_pie = [self.COLORS["accent"], self.COLORS["primary"]]
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=["Normal (0)", "Anomali (1)"],
            colors=colors_pie,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"color": self.COLORS["text"], "fontsize": 12},
            wedgeprops={"edgecolor": self.COLORS["bg_dark"], "linewidth": 2},
            explode=(0, 0.05),
        )
        for t in autotexts:
            t.set_fontweight("bold")
        ax.set_title("Sınıf Dağılımı", color=self.COLORS["secondary"],
                      fontsize=14, weight="bold")
        return fig

    def plot_cell_type_distribution(self, df: pd.DataFrame, col: str = "cell_type"):
        """Hücre tipi dağılımı bar chart."""
        fig, ax = plt.subplots(figsize=(10, 5))
        self._apply_style(fig, ax)

        ct_counts = df[col].value_counts()
        norm_vals = ct_counts.values / ct_counts.values.max()
        colors = [plt.cm.magma(0.3 + 0.5 * v) for v in norm_vals]

        ax.bar(range(len(ct_counts)), ct_counts.values, color=colors,
               edgecolor="white", linewidth=0.3, alpha=0.9)
        ax.set_xticks(range(len(ct_counts)))
        ax.set_xticklabels(ct_counts.index, rotation=45, ha="right",
                           fontsize=8, color=self.COLORS["text"])
        ax.set_ylabel("Sayı", color=self.COLORS["text"])
        ax.set_title("Hücre Tipi Dağılımı", color=self.COLORS["secondary"],
                      fontsize=14, weight="bold", pad=10)
        ax.grid(True, axis="y", alpha=0.1)
        plt.tight_layout()
        return fig

    def plot_feature_histogram(self, df: pd.DataFrame, feature: str,
                                target_col: str = "anomaly_label"):
        """Normal vs Anomali özellik dağılımı histogram."""
        fig, ax = plt.subplots(figsize=(10, 4))
        self._apply_style(fig, ax)

        for label, color, lbl_text in [
            (0, self.COLORS["accent"], "Normal"),
            (1, self.COLORS["primary"], "Anomali"),
        ]:
            subset = df[df[target_col] == label][feature]
            ax.hist(subset, bins=40, alpha=0.6, color=color, label=lbl_text,
                    edgecolor="white", linewidth=0.2)

        ax.set_xlabel(feature, color=self.COLORS["text"], fontsize=11)
        ax.set_ylabel("Frekans", color=self.COLORS["text"], fontsize=11)
        ax.set_title(f"{feature} Dağılımı (Normal vs Anomali)",
                      color=self.COLORS["secondary"], fontsize=13, weight="bold", pad=10)
        ax.legend(facecolor=self.COLORS["bg_card"], edgecolor="#333",
                  labelcolor=self.COLORS["text"])
        ax.grid(True, axis="y", alpha=0.1)
        plt.tight_layout()
        return fig

    def plot_correlation_matrix(self, df: pd.DataFrame):
        """Korelasyon matrisi heatmap."""
        corr = df.select_dtypes(include=["number"]).corr()
        fig, ax = plt.subplots(figsize=(14, 10))
        self._apply_style(fig, ax)

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, linecolor=self.COLORS["bg_dark"],
            ax=ax, annot_kws={"size": 6}, square=True,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Özellik Korelasyon Matrisi", color=self.COLORS["secondary"],
                      fontsize=14, weight="bold", pad=15)
        ax.tick_params(colors=self.COLORS["text"], labelsize=7)
        plt.tight_layout()
        return fig

    def plot_box_anomaly_comparison(self, df: pd.DataFrame, features: list,
                                     target_col: str = "anomaly_label"):
        """Normal vs Anomali box plot karşılaştırması (seçili özellikler)."""
        n = len(features)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        fig.patch.set_facecolor(self.COLORS["bg_dark"])

        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, feat in enumerate(features):
            ax = axes[i]
            ax.set_facecolor(self.COLORS["bg_card"])

            data_normal = df[df[target_col] == 0][feat].dropna()
            data_anomaly = df[df[target_col] == 1][feat].dropna()

            bp = ax.boxplot(
                [data_normal, data_anomaly],
                labels=["Normal", "Anomali"],
                patch_artist=True,
                boxprops={"facecolor": self.COLORS["accent"], "alpha": 0.6},
                medianprops={"color": self.COLORS["secondary"], "linewidth": 2},
                whiskerprops={"color": self.COLORS["text_muted"]},
                capprops={"color": self.COLORS["text_muted"]},
                flierprops={"markerfacecolor": self.COLORS["primary"], "markersize": 3},
            )
            bp["boxes"][1].set_facecolor(self.COLORS["primary"])

            ax.set_title(feat, color=self.COLORS["secondary"], fontsize=10, weight="bold")
            ax.tick_params(colors=self.COLORS["text"], labelsize=8)
            ax.grid(True, axis="y", alpha=0.1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(self.COLORS["text_muted"])
            ax.spines["bottom"].set_color(self.COLORS["text_muted"])
            ax.spines["left"].set_alpha(0.3)
            ax.spines["bottom"].set_alpha(0.3)

        # Boş axes'i gizle
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Normal vs Anomali Karşılaştırması",
                      color=self.COLORS["secondary"], fontsize=14, weight="bold", y=1.02)
        plt.tight_layout()
        return fig
