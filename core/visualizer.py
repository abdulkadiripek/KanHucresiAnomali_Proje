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
    # EDA Grafikleri — Hikaye anlatan, otomatik seçimli (kullanıcı seçmez)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_cohens_d(group1: pd.Series, group2: pd.Series) -> float:
        """Cohen's d effect size — iki grup ortalaması arasındaki standardize fark.

        |d| < 0.2 ihmal edilebilir, 0.2-0.5 küçük, 0.5-0.8 orta, > 0.8 büyük etki.
        """
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0.0
        s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
        pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
        if pooled == 0 or np.isnan(pooled):
            return 0.0
        return (group1.mean() - group2.mean()) / pooled

    def plot_leakage_evidence(
        self,
        raw_df: pd.DataFrame,
        target_col: str = "anomaly_label",
        leakage_cols: Optional[list] = None,
    ):
        """Tüm sayısal sütunların hedefle mutlak korelasyonu — leakage atma gerekçesi.

        Atılan sütunlar kırmızı, modele giren sütunlar yeşil tonda.
        """
        leakage_cols = leakage_cols or []
        num_df = raw_df.select_dtypes(include=["number"]).copy()
        if target_col not in num_df.columns:
            num_df[target_col] = raw_df[target_col]

        corr = num_df.corr()[target_col].drop(target_col).abs().sort_values(ascending=True)

        colors = [
            self.COLORS["primary"] if col in leakage_cols else self.COLORS["success"]
            for col in corr.index
        ]

        fig, ax = plt.subplots(figsize=(10, max(5, len(corr) * 0.28)))
        self._apply_style(fig, ax)

        ax.barh(range(len(corr)), corr.values, color=colors,
                edgecolor="white", linewidth=0.3, alpha=0.92)
        ax.set_yticks(range(len(corr)))
        ax.set_yticklabels(corr.index, fontsize=8.5, color=self.COLORS["text"])
        ax.set_xlabel("|Pearson Korelasyon| (anomaly_label ile)",
                      color=self.COLORS["text"], fontsize=11)
        ax.set_title(
            "Veri Sızıntısı (Data Leakage) Kanıtı\n"
            "Kırmızı: Hedefi ele veren sütunlar (atıldı)  |  Yeşil: Modele giren gerçek özellikler",
            color=self.COLORS["secondary"], fontsize=13, weight="bold", pad=12,
        )
        ax.axvline(x=0.5, color=self.COLORS["secondary"], linestyle="--",
                   alpha=0.4, linewidth=1)
        ax.text(0.51, 0.5, "Şüpheli eşik (0.5)", color=self.COLORS["secondary"],
                fontsize=8, alpha=0.7, transform=ax.get_yaxis_transform())
        ax.grid(True, axis="x", alpha=0.1)
        plt.tight_layout()
        return fig

    def plot_class_imbalance_donut(self, labels: pd.Series):
        """Sınıf dengesizliği donut chart — ortasında oran metni."""
        counts = labels.value_counts().sort_index()
        n_normal = int(counts.get(0, 0))
        n_anomaly = int(counts.get(1, 0))
        total = n_normal + n_anomaly
        anomaly_pct = (n_anomaly / total * 100) if total else 0

        fig, ax = plt.subplots(figsize=(7, 5.5))
        fig.patch.set_facecolor(self.COLORS["bg_dark"])

        colors_donut = [self.COLORS["accent"], self.COLORS["primary"]]
        wedges, _ = ax.pie(
            [n_normal, n_anomaly],
            colors=colors_donut,
            startangle=90,
            wedgeprops={"width": 0.38, "edgecolor": self.COLORS["bg_dark"], "linewidth": 3},
        )

        # Ortadaki metin
        ax.text(0, 0.08, f"%{anomaly_pct:.1f}",
                ha="center", va="center", fontsize=32, weight="bold",
                color=self.COLORS["primary"])
        ax.text(0, -0.18, "Anomali", ha="center", va="center",
                fontsize=11, color=self.COLORS["text_muted"], weight="500")

        # Legend
        ax.legend(
            wedges,
            [f"Normal  ·  {n_normal:,}", f"Anomali  ·  {n_anomaly:,}"],
            loc="center left", bbox_to_anchor=(1.0, 0.5),
            fontsize=10, facecolor=self.COLORS["bg_card"],
            edgecolor="#333", labelcolor=self.COLORS["text"],
            frameon=True,
        )

        ax.set_title("Sınıf Dengesizliği",
                     color=self.COLORS["secondary"], fontsize=14, weight="bold", pad=15)
        plt.tight_layout()
        return fig

    def plot_cohens_d_top_features(
        self,
        df_clean: pd.DataFrame,
        target_col: str = "anomaly_label",
        top_n: int = 10,
    ) -> tuple:
        """Cohen's d ile sınıflar arası en ayırt edici özellikler.

        Returns:
            (fig, top_features_list)
        """
        num_df = df_clean.select_dtypes(include=["number"]).drop(columns=[target_col], errors="ignore")
        y = df_clean[target_col]

        d_values = {}
        for col in num_df.columns:
            g0 = num_df.loc[y == 0, col].dropna()
            g1 = num_df.loc[y == 1, col].dropna()
            d_values[col] = abs(self.compute_cohens_d(g1, g0))

        d_series = pd.Series(d_values).sort_values(ascending=True)
        top_series = d_series.tail(top_n)

        # Etki büyüklüğüne göre renklendir
        def _effect_color(d):
            if d >= 0.8:
                return self.COLORS["primary"]    # büyük
            elif d >= 0.5:
                return self.COLORS["secondary"]  # orta
            elif d >= 0.2:
                return self.COLORS["accent"]     # küçük
            return self.COLORS["text_muted"]     # ihmal edilebilir

        colors = [_effect_color(v) for v in top_series.values]

        fig, ax = plt.subplots(figsize=(10, max(4.5, top_n * 0.5)))
        self._apply_style(fig, ax)

        bars = ax.barh(range(len(top_series)), top_series.values, color=colors,
                       edgecolor="white", linewidth=0.3, alpha=0.92)
        ax.set_yticks(range(len(top_series)))
        ax.set_yticklabels(top_series.index, fontsize=10, color=self.COLORS["text"])
        ax.set_xlabel("|Cohen's d| (Etki Büyüklüğü)",
                      color=self.COLORS["text"], fontsize=11)
        ax.set_title(
            f"Top {top_n} Ayırt Edici Özellik — Cohen's d Etki Büyüklüğü\n"
            "Modelin en güçlü sinyalleri buradan geliyor",
            color=self.COLORS["secondary"], fontsize=13, weight="bold", pad=12,
        )

        # Eşik çizgileri
        for thr, label in [(0.2, "Küçük"), (0.5, "Orta"), (0.8, "Büyük")]:
            ax.axvline(x=thr, color=self.COLORS["text_muted"],
                       linestyle="--", alpha=0.3, linewidth=1)
            ax.text(thr, len(top_series) - 0.4, label, fontsize=8,
                    color=self.COLORS["text_muted"], ha="center", alpha=0.7)

        # Bar üstü değer
        for bar, val in zip(bars, top_series.values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=8.5,
                    color=self.COLORS["text"], weight="bold")

        ax.grid(True, axis="x", alpha=0.1)
        plt.tight_layout()

        top_features = list(top_series.index[::-1])  # en güçlüden zayıfa
        return fig, top_features

    def plot_top_features_kde(
        self,
        df_clean: pd.DataFrame,
        features: list,
        target_col: str = "anomaly_label",
    ):
        """Top N özellik için Normal vs Anomali yoğunluk (KDE) dağılımı.

        2x2 (4 özellik) veya 1x3 (3 özellik) grid otomatik.
        """
        from scipy.stats import gaussian_kde

        n = len(features)
        if n == 0:
            return None

        if n <= 2:
            rows, cols = 1, n
        elif n <= 4:
            rows, cols = 2, 2
        else:
            cols = 3
            rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        fig.patch.set_facecolor(self.COLORS["bg_dark"])

        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, feat in enumerate(features):
            ax = axes[i]
            ax.set_facecolor(self.COLORS["bg_card"])

            data_normal = df_clean.loc[df_clean[target_col] == 0, feat].dropna().values
            data_anomaly = df_clean.loc[df_clean[target_col] == 1, feat].dropna().values

            if len(data_normal) < 2 or len(data_anomaly) < 2:
                ax.set_visible(False)
                continue

            x_min = min(data_normal.min(), data_anomaly.min())
            x_max = max(data_normal.max(), data_anomaly.max())
            x_grid = np.linspace(x_min, x_max, 250)

            try:
                kde_n = gaussian_kde(data_normal)
                kde_a = gaussian_kde(data_anomaly)
                y_n = kde_n(x_grid)
                y_a = kde_a(x_grid)

                ax.fill_between(x_grid, y_n, color=self.COLORS["accent"],
                                alpha=0.45, label="Normal")
                ax.fill_between(x_grid, y_a, color=self.COLORS["primary"],
                                alpha=0.45, label="Anomali")
                ax.plot(x_grid, y_n, color=self.COLORS["accent"], linewidth=1.5)
                ax.plot(x_grid, y_a, color=self.COLORS["primary"], linewidth=1.5)
            except Exception:
                # KDE hata verirse histograma düş
                ax.hist(data_normal, bins=30, alpha=0.5, color=self.COLORS["accent"],
                        label="Normal", density=True)
                ax.hist(data_anomaly, bins=30, alpha=0.5, color=self.COLORS["primary"],
                        label="Anomali", density=True)

            d_val = abs(self.compute_cohens_d(
                pd.Series(data_anomaly), pd.Series(data_normal)
            ))
            ax.set_title(f"{feat}   (|d| = {d_val:.2f})",
                         color=self.COLORS["secondary"], fontsize=11, weight="bold")
            ax.set_xlabel(feat, color=self.COLORS["text"], fontsize=9)
            ax.set_ylabel("Yoğunluk", color=self.COLORS["text"], fontsize=9)
            ax.tick_params(colors=self.COLORS["text"], labelsize=8)
            ax.legend(fontsize=8, facecolor=self.COLORS["bg_card"],
                      edgecolor="#333", labelcolor=self.COLORS["text"])
            ax.grid(True, alpha=0.1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(self.COLORS["text_muted"])
            ax.spines["bottom"].set_color(self.COLORS["text_muted"])
            ax.spines["left"].set_alpha(0.3)
            ax.spines["bottom"].set_alpha(0.3)

        # Boş axes
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            "En Ayırt Edici Özelliklerde Normal vs Anomali Dağılımı",
            color=self.COLORS["secondary"], fontsize=14, weight="bold", y=1.00,
        )
        plt.tight_layout()
        return fig

    def plot_categorical_bias(
        self,
        df: pd.DataFrame,
        cat_cols: list,
        target_col: str = "anomaly_label",
    ):
        """Kategorik sütunlarda anomali oranı — bias kontrolü.

        Her kategori için anomali oranı bar chart. Genel anomali oranı
        baseline çizgisi olarak gösterilir. Sapma > 5 puansa kategori vurgulanır.
        """
        baseline = df[target_col].mean()
        n = len(cat_cols)
        if n == 0:
            return None

        cols = 4 if n >= 4 else n
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        fig.patch.set_facecolor(self.COLORS["bg_dark"])

        if n == 1:
            axes = np.array([axes])
        axes = np.array(axes).flatten()

        for i, col in enumerate(cat_cols):
            ax = axes[i]
            ax.set_facecolor(self.COLORS["bg_card"])

            grp = df.groupby(col)[target_col].agg(["mean", "count"]).sort_values("mean")
            categories = [str(c) for c in grp.index]
            rates = grp["mean"].values
            counts = grp["count"].values

            # Renk: baseline'dan sapma
            colors = []
            for r in rates:
                deviation = r - baseline
                if deviation > 0.05:
                    colors.append(self.COLORS["primary"])      # Yüksek anomali oranı
                elif deviation < -0.05:
                    colors.append(self.COLORS["accent"])       # Düşük anomali oranı
                else:
                    colors.append(self.COLORS["text_muted"])   # Baseline civarı

            bars = ax.bar(range(len(categories)), rates, color=colors,
                          edgecolor="white", linewidth=0.4, alpha=0.92)

            # Baseline çizgisi
            ax.axhline(y=baseline, color=self.COLORS["secondary"],
                       linestyle="--", linewidth=1.5, alpha=0.85,
                       label=f"Genel ort. ({baseline:.0%})")

            # Bar üstü oran ve n
            for bar, r, n_count in zip(bars, rates, counts):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.015,
                        f"{r:.0%}",
                        ha="center", va="bottom", fontsize=10, weight="bold",
                        color=self.COLORS["text"])
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() / 2,
                        f"n={n_count:,}",
                        ha="center", va="center", fontsize=8,
                        color="white", alpha=0.7)

            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=20, ha="right",
                               fontsize=9, color=self.COLORS["text"])
            ax.set_ylabel("Anomali Oranı", color=self.COLORS["text"], fontsize=9)
            ax.set_ylim(0, max(max(rates) * 1.25, baseline * 1.4, 0.3))
            ax.set_title(col, color=self.COLORS["secondary"],
                         fontsize=11, weight="bold", pad=8)
            ax.tick_params(colors=self.COLORS["text"], labelsize=8)
            ax.legend(fontsize=7, loc="upper left",
                      facecolor=self.COLORS["bg_card"], edgecolor="#333",
                      labelcolor=self.COLORS["text"])
            ax.grid(True, axis="y", alpha=0.1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(self.COLORS["text_muted"])
            ax.spines["bottom"].set_color(self.COLORS["text_muted"])
            ax.spines["left"].set_alpha(0.3)
            ax.spines["bottom"].set_alpha(0.3)

        # Boş axes'i gizle
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            "Kategorik Özelliklerde Anomali Oranı — Bias Kontrolü\n"
            "Kırmızı: yüksek anomali  ·  Mavi: düşük anomali  ·  Gri: baseline civarı",
            color=self.COLORS["secondary"], fontsize=13, weight="bold", y=1.00,
        )
        plt.tight_layout()
        return fig

    @staticmethod
    def compute_categorical_bias_summary(
        df: pd.DataFrame,
        cat_cols: list,
        target_col: str = "anomaly_label",
    ) -> pd.DataFrame:
        """Her kategorik sütun için max-min anomali oranı farkını hesapla.

        Yüksek fark = bu kategorik sütun anomali ile güçlü bir ilişkiye sahip
        (klinik anlamlı veya bias göstergesi olabilir).
        """
        rows = []
        baseline = df[target_col].mean()
        for col in cat_cols:
            grp = df.groupby(col)[target_col].agg(["mean", "count"])
            rate_max = grp["mean"].max()
            rate_min = grp["mean"].min()
            spread = rate_max - rate_min
            top_cat = grp["mean"].idxmax()
            bot_cat = grp["mean"].idxmin()
            rows.append({
                "Sütun": col,
                "Yelpaze (max-min)": spread,
                "En Yüksek": f"{top_cat} ({rate_max:.1%})",
                "En Düşük": f"{bot_cat} ({rate_min:.1%})",
                "Baseline": f"{baseline:.1%}",
            })
        return pd.DataFrame(rows).sort_values("Yelpaze (max-min)", ascending=False)
