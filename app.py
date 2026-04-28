"""
=============================================================================
🩸 Blood Cell Anomaly Detection — Streamlit Web Arayüzü
=============================================================================
Klasik Makine Öğrenmesi algoritmaları (XGBoost, Random Forest, LightGBM)
kullanarak kan hücresi anomalilerini tespit eden interaktif dashboard.

Çalıştırmak için:  streamlit run app.py
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Uyarıları bastır
warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ============================================================================
# SAYFA AYARLARI
# ============================================================================
st.set_page_config(
    page_title="🩸 Kan Hücresi Anomali Tespiti",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS — Premium Dark Theme
# ============================================================================
st.markdown("""
<style>
    /* Ana arka plan */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* Metrik kartları */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(26,26,46,0.8), rgba(22,33,62,0.8));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.5);
    }

    /* Başlık stillleri */
    h1 {
        background: linear-gradient(90deg, #e94560, #ff6b6b, #feca57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    h2 {
        color: #e94560 !important;
        border-bottom: 2px solid rgba(233,69,96,0.3);
        padding-bottom: 8px;
    }
    h3 {
        color: #feca57 !important;
    }

    /* Tablar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(15,12,41,0.5);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        color: #a0a0b8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #e94560, #ff6b6b) !important;
        color: white !important;
    }

    /* Buton */
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c0392b) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 12px 32px !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(233,69,96,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(233,69,96,0.5) !important;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Divider */
    hr {
        border-color: rgba(233,69,96,0.2) !important;
    }

    /* Info/Success/Warning kutuları */
    .stAlert {
        border-radius: 10px !important;
    }

    /* Slider */
    .stSlider > div > div > div {
        background: #e94560 !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================
@st.cache_data
def load_data():
    """Veri setini yükle ve döndür."""
    data_path = os.path.join(os.path.dirname(__file__), "archive", "blood_cell_anomaly_detection.csv")
    return pd.read_csv(data_path)


def clean_data(df):
    """Data Leakage'a neden olacak ve gereksiz sütunları kaldır."""
    drop_cols = [
        "cell_id",
        "disease_category",
        "cell_type",
        "cytodiffusion_anomaly_score",
        "cytodiffusion_classification_confidence",
        "labeller_confidence_score",
    ]
    return df.drop(columns=drop_cols)


def preprocess(df, test_size=0.20, random_state=42):
    """Ön işleme: encoding, scaling, train/test split."""
    y = df["anomaly_label"]
    X = df.drop(columns=["anomaly_label"])

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()

    # One-Hot Encoding
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scaling (sadece sayısal sütunlar)
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train, X_test, y_train, y_test, numerical_cols, categorical_cols


def get_model(name, params):
    """Seçilen algoritmaya göre model döndür."""
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    elif name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"] if params["max_depth"] > 0 else None,
            min_samples_split=params.get("min_samples_split", 5),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            random_state=42,
            n_jobs=-1,
        )
    elif name == "LightGBM":
        return LGBMClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"] if params["max_depth"] > 0 else -1,
            learning_rate=params["learning_rate"],
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            random_state=42,
            verbose=-1,
        )


def plot_confusion_matrix(y_true, y_pred, model_name, f1):
    """Confusion Matrix görselleştirmesi."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor("#0f0c29")
    ax.set_facecolor("#1a1a2e")

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="magma",
        xticklabels=["Normal (0)", "Anomali (1)"],
        yticklabels=["Normal (0)", "Anomali (1)"],
        linewidths=2,
        linecolor="#0f0c29",
        annot_kws={"size": 20, "weight": "bold", "color": "white"},
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Tahmin Edilen", fontsize=13, color="white", labelpad=10)
    ax.set_ylabel("Gerçek", fontsize=13, color="white", labelpad=10)
    ax.set_title(
        f"Confusion Matrix — {model_name}\nF1-Score: {f1:.4f}",
        fontsize=14,
        weight="bold",
        color="#feca57",
        pad=15,
    )
    ax.tick_params(colors="white")
    plt.tight_layout()
    return fig


def plot_roc_curves(results_dict, y_test):
    """Tüm modellerin ROC Curve'lerini çiz."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f0c29")
    ax.set_facecolor("#1a1a2e")

    colors = ["#e94560", "#feca57", "#00d2ff"]
    for i, (name, data) in enumerate(results_dict.items()):
        if "y_proba" in data and data["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, data["y_proba"])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2.5,
                    label=f"{name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate", color="white", fontsize=12)
    ax.set_ylabel("True Positive Rate", color="white", fontsize=12)
    ax.set_title("ROC Curve Karşılaştırması", color="#feca57", fontsize=14, weight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=10, facecolor="#1a1a2e", edgecolor="#333",
              labelcolor="white")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, model_name, top_n=15):
    """Feature importance bar chart."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#0f0c29")
        ax.set_facecolor("#1a1a2e")

        bars = ax.barh(range(len(top_features)), top_importances, color="#e94560", edgecolor="#ff6b6b", linewidth=0.5)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=9, color="white")
        ax.set_xlabel("Önem Skoru", color="white", fontsize=12)
        ax.set_title(f"Top {top_n} Özellik — {model_name}", color="#feca57", fontsize=14, weight="bold", pad=15)
        ax.tick_params(colors="white")
        ax.grid(True, axis="x", alpha=0.15)
        plt.tight_layout()
        return fig
    return None


def plot_metric_comparison(results_dict):
    """Model karşılaştırma bar chart."""
    metrics_df = pd.DataFrame({
        name: {k: v for k, v in data.items() if k in ["Accuracy", "Precision", "Recall", "F1-Score"]}
        for name, data in results_dict.items()
    }).T

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f0c29")
    ax.set_facecolor("#1a1a2e")

    x = np.arange(len(metrics_df.index))
    width = 0.18
    colors = ["#e94560", "#feca57", "#00d2ff", "#7bed9f"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]

    for i, metric in enumerate(metric_names):
        bars = ax.bar(x + i * width, metrics_df[metric], width, label=metric,
                      color=colors[i], edgecolor="white", linewidth=0.3, alpha=0.9)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=7,
                    color="white", weight="bold")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics_df.index, fontsize=11, color="white")
    ax.set_ylabel("Skor", color="white", fontsize=12)
    ax.set_title("Model Performans Karşılaştırması", color="#feca57", fontsize=14, weight="bold", pad=15)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white", loc="upper left")
    ax.tick_params(colors="white")
    ax.grid(True, axis="y", alpha=0.15)
    plt.tight_layout()
    return fig


# ============================================================================
# ANA UYGULAMA
# ============================================================================
def main():
    # ------ SIDEBAR ------
    with st.sidebar:
        st.markdown("## 🔬 Kontrol Paneli")
        st.markdown("---")

        st.markdown("### 📊 Veri Ayarları")
        test_size = st.slider(
            "Test Seti Oranı (%)", min_value=10, max_value=40, value=20, step=5,
            help="Verinin yüzde kaçı test için ayrılsın?"
        ) / 100.0

        st.markdown("---")

        st.markdown("### 🤖 Model Seçimi")
        selected_models = st.multiselect(
            "Eğitilecek Modeller",
            ["XGBoost", "Random Forest", "LightGBM"],
            default=["XGBoost", "Random Forest", "LightGBM"],
            help="En az 1 model seçmelisiniz."
        )

        st.markdown("---")

        # Hiperparametre ayarları
        st.markdown("### ⚙️ Hiperparametreler")
        model_params = {}

        if "XGBoost" in selected_models:
            with st.expander("🔷 XGBoost Parametreleri", expanded=False):
                model_params["XGBoost"] = {
                    "n_estimators": st.slider("n_estimators (XGB)", 50, 1000, 300, 50, key="xgb_n"),
                    "max_depth": st.slider("max_depth (XGB)", 2, 15, 6, 1, key="xgb_d"),
                    "learning_rate": st.slider("learning_rate (XGB)", 0.01, 0.5, 0.1, 0.01, key="xgb_lr"),
                    "subsample": st.slider("subsample (XGB)", 0.5, 1.0, 0.8, 0.05, key="xgb_ss"),
                    "colsample_bytree": st.slider("colsample_bytree (XGB)", 0.5, 1.0, 0.8, 0.05, key="xgb_cs"),
                }

        if "Random Forest" in selected_models:
            with st.expander("🌲 Random Forest Parametreleri", expanded=False):
                model_params["Random Forest"] = {
                    "n_estimators": st.slider("n_estimators (RF)", 50, 1000, 300, 50, key="rf_n"),
                    "max_depth": st.slider("max_depth (RF, 0=None)", 0, 30, 0, 1, key="rf_d"),
                    "min_samples_split": st.slider("min_samples_split (RF)", 2, 20, 5, 1, key="rf_mss"),
                    "min_samples_leaf": st.slider("min_samples_leaf (RF)", 1, 10, 2, 1, key="rf_msl"),
                }

        if "LightGBM" in selected_models:
            with st.expander("💡 LightGBM Parametreleri", expanded=False):
                model_params["LightGBM"] = {
                    "n_estimators": st.slider("n_estimators (LGBM)", 50, 1000, 300, 50, key="lgb_n"),
                    "max_depth": st.slider("max_depth (LGBM, 0=Auto)", 0, 15, 0, 1, key="lgb_d"),
                    "learning_rate": st.slider("learning_rate (LGBM)", 0.01, 0.5, 0.1, 0.01, key="lgb_lr"),
                    "subsample": st.slider("subsample (LGBM)", 0.5, 1.0, 0.8, 0.05, key="lgb_ss"),
                    "colsample_bytree": st.slider("colsample_bytree (LGBM)", 0.5, 1.0, 0.8, 0.05, key="lgb_cs"),
                }

        st.markdown("---")
        train_button = st.button("🚀 Modelleri Eğit", use_container_width=True, type="primary")

    # ------ ANA İÇERİK ------
    st.markdown("# 🩸 Kan Hücresi Anomali Tespiti")
    st.markdown(
        "> **Klasik Makine Öğrenmesi** ile kan hücrelerindeki anomalileri tespit eden "
        "interaktif dashboard. Derin öğrenme **kullanılmamıştır**."
    )

    # Veri yükle
    raw_df = load_data()

    # ---- TABS ----
    tab_data, tab_eda, tab_train, tab_results = st.tabs([
        "📋 Veri Seti", "📊 Keşifsel Analiz (EDA)", "🤖 Eğitim", "📈 Sonuçlar"
    ])

    # ============================================================
    # TAB 1 — VERİ SETİ
    # ============================================================
    with tab_data:
        st.markdown("## 📋 Veri Seti Genel Bakış")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Toplam Satır", f"{raw_df.shape[0]:,}")
        col2.metric("Toplam Sütun", f"{raw_df.shape[1]}")
        col3.metric("Anomali Sayısı", f"{raw_df['anomaly_label'].sum():,}")
        col4.metric("Anomali Oranı", f"{raw_df['anomaly_label'].mean():.1%}")

        st.markdown("### 🔍 İlk 10 Satır")
        st.dataframe(raw_df.head(10), use_container_width=True, height=400)

        st.markdown("### 📊 Veri Tipleri ve Eksik Değerler")
        info_df = pd.DataFrame({
            "Sütun": raw_df.columns,
            "Tip": raw_df.dtypes.astype(str).values,
            "Eksik": raw_df.isnull().sum().values,
            "Benzersiz": raw_df.nunique().values,
        })
        st.dataframe(info_df, use_container_width=True, height=400)

        st.markdown("### 🧹 Kaldırılacak Sütunlar (Data Leakage Önlemi)")
        st.warning(
            "Aşağıdaki sütunlar Data Leakage'a neden olacağı için model eğitiminden **çıkarılır**:\n\n"
            "- `cell_id` — Kimlik bilgisi, modele katkısı yok\n"
            "- `disease_category` — Doğrudan hedef ile ilişkili\n"
            "- `cell_type` — Doğrudan hedef ile ilişkili\n"
            "- `cytodiffusion_anomaly_score` — Anomali bilgisi sızıntısı\n"
            "- `cytodiffusion_classification_confidence` — Sınıflandırma bilgisi sızıntısı\n"
            "- `labeller_confidence_score` — Etiketleyici güven skoru sızıntısı"
        )

    # ============================================================
    # TAB 2 — EDA
    # ============================================================
    with tab_eda:
        st.markdown("## 📊 Keşifsel Veri Analizi (EDA)")

        df_clean = clean_data(raw_df.copy())

        # Anomali dağılımı
        st.markdown("### 🎯 Anomali Dağılımı")
        col_a, col_b = st.columns([1, 2])

        with col_a:
            anomaly_counts = raw_df["anomaly_label"].value_counts()
            fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
            fig_pie.patch.set_facecolor("#0f0c29")
            colors_pie = ["#00d2ff", "#e94560"]
            wedges, texts, autotexts = ax_pie.pie(
                anomaly_counts.values,
                labels=["Normal (0)", "Anomali (1)"],
                colors=colors_pie,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"color": "white", "fontsize": 12},
                wedgeprops={"edgecolor": "#0f0c29", "linewidth": 2},
                explode=(0, 0.05),
            )
            for t in autotexts:
                t.set_fontweight("bold")
            ax_pie.set_title("Sınıf Dağılımı", color="#feca57", fontsize=14, weight="bold")
            st.pyplot(fig_pie)
            plt.close()

        with col_b:
            # Hücre tipi dağılımı
            fig_ct, ax_ct = plt.subplots(figsize=(10, 5))
            fig_ct.patch.set_facecolor("#0f0c29")
            ax_ct.set_facecolor("#1a1a2e")
            ct_counts = raw_df["cell_type"].value_counts()
            bars = ax_ct.bar(range(len(ct_counts)), ct_counts.values, color="#e94560",
                             edgecolor="#ff6b6b", linewidth=0.5, alpha=0.9)
            ax_ct.set_xticks(range(len(ct_counts)))
            ax_ct.set_xticklabels(ct_counts.index, rotation=45, ha="right", fontsize=8, color="white")
            ax_ct.set_ylabel("Sayı", color="white")
            ax_ct.set_title("Hücre Tipi Dağılımı", color="#feca57", fontsize=14, weight="bold", pad=10)
            ax_ct.tick_params(colors="white")
            ax_ct.grid(True, axis="y", alpha=0.15)
            plt.tight_layout()
            st.pyplot(fig_ct)
            plt.close()

        # Sayısal değişkenlerin dağılımı
        st.markdown("### 📈 Sayısal Özellik Dağılımları")
        num_cols = df_clean.select_dtypes(include=["number"]).columns.drop("anomaly_label").tolist()

        selected_feat = st.selectbox("Özellik Seçin:", num_cols, index=0)
        fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
        fig_hist.patch.set_facecolor("#0f0c29")
        ax_hist.set_facecolor("#1a1a2e")

        for label, color, lbl_text in [(0, "#00d2ff", "Normal"), (1, "#e94560", "Anomali")]:
            subset = df_clean[df_clean["anomaly_label"] == label][selected_feat]
            ax_hist.hist(subset, bins=40, alpha=0.6, color=color, label=lbl_text, edgecolor="white", linewidth=0.3)

        ax_hist.set_xlabel(selected_feat, color="white", fontsize=11)
        ax_hist.set_ylabel("Frekans", color="white", fontsize=11)
        ax_hist.set_title(f"{selected_feat} Dağılımı (Normal vs Anomali)", color="#feca57",
                          fontsize=13, weight="bold", pad=10)
        ax_hist.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
        ax_hist.tick_params(colors="white")
        ax_hist.grid(True, axis="y", alpha=0.15)
        plt.tight_layout()
        st.pyplot(fig_hist)
        plt.close()

        # Korelasyon matrisi
        st.markdown("### 🔗 Korelasyon Matrisi")
        corr_matrix = df_clean.select_dtypes(include=["number"]).corr()
        fig_corr, ax_corr = plt.subplots(figsize=(14, 10))
        fig_corr.patch.set_facecolor("#0f0c29")
        ax_corr.set_facecolor("#1a1a2e")
        sns.heatmap(
            corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, linecolor="#0f0c29", ax=ax_corr,
            annot_kws={"size": 6}, square=True,
            cbar_kws={"shrink": 0.8},
        )
        ax_corr.set_title("Özellik Korelasyon Matrisi", color="#feca57", fontsize=14, weight="bold", pad=15)
        ax_corr.tick_params(colors="white", labelsize=7)
        plt.tight_layout()
        st.pyplot(fig_corr)
        plt.close()

    # ============================================================
    # TAB 3 — EĞİTİM
    # ============================================================
    with tab_train:
        st.markdown("## 🤖 Model Eğitimi")

        if not selected_models:
            st.error("⚠️ Lütfen sol panelden en az 1 model seçin.")
            return

        if not train_button:
            st.info(
                "👈 Sol paneldeki **kontrolleri** ayarlayın ve **'🚀 Modelleri Eğit'** "
                "butonuna tıklayın."
            )

            st.markdown("### 📝 Eğitim Süreci Adımları")
            st.markdown("""
            1. **Veri Temizleme** — Gereksiz ve leakage sütunları kaldırılır
            2. **One-Hot Encoding** — Kategorik değişkenler sayısallaştırılır
            3. **Train/Test Split** — Veri stratified olarak bölünür
            4. **StandardScaler** — Sayısal özellikler ölçeklenir (fit → train, transform → test)
            5. **Model Eğitimi** — Seçilen algoritmalar eğitilir
            6. **Değerlendirme** — Metrikler ve confusion matrix üretilir
            """)
            return

        # ---- EĞİTİM BAŞLAT ----
        progress = st.progress(0, text="Veri hazırlanıyor...")

        # Veri temizle ve ön işle
        df_clean = clean_data(raw_df.copy())
        progress.progress(10, text="Veri temizlendi. Ön işleme yapılıyor...")

        X_train, X_test, y_train, y_test, numerical_cols, categorical_cols = preprocess(
            df_clean, test_size=test_size
        )
        progress.progress(25, text="Ön işleme tamamlandı. Modeller eğitiliyor...")

        st.success(
            f"✅ Veri hazır — Train: {X_train.shape[0]} satır, Test: {X_test.shape[0]} satır, "
            f"Özellik sayısı: {X_train.shape[1]}"
        )

        # Model eğitimi
        results_dict = {}
        trained_models = {}
        feature_names = X_train.columns.tolist()

        for i, name in enumerate(selected_models):
            params = model_params.get(name, {
                "n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "min_samples_split": 5, "min_samples_leaf": 2,
            })

            pct = 25 + int((i / len(selected_models)) * 60)
            progress.progress(pct, text=f"{name} eğitiliyor...")

            model = get_model(name, params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Olasılık tahmini (ROC Curve için)
            y_proba = None
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            results_dict[name] = {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "y_pred": y_pred,
                "y_proba": y_proba,
            }
            trained_models[name] = model

        progress.progress(100, text="✅ Tüm modeller eğitildi!")

        # Sonuçları session_state'e kaydet
        st.session_state["results_dict"] = results_dict
        st.session_state["trained_models"] = trained_models
        st.session_state["y_test"] = y_test
        st.session_state["feature_names"] = feature_names
        st.session_state["trained"] = True

        # Hızlı özet
        st.markdown("### 📊 Eğitim Özeti")
        for name, metrics in results_dict.items():
            with st.expander(f"✅ {name}", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                c2.metric("Precision", f"{metrics['Precision']:.4f}")
                c3.metric("Recall", f"{metrics['Recall']:.4f}")
                c4.metric("F1-Score", f"{metrics['F1-Score']:.4f}")

    # ============================================================
    # TAB 4 — SONUÇLAR
    # ============================================================
    with tab_results:
        st.markdown("## 📈 Detaylı Sonuçlar ve Karşılaştırma")

        if "trained" not in st.session_state or not st.session_state["trained"]:
            st.info("⏳ Henüz model eğitilmedi. 'Eğitim' sekmesinden modelleri eğitin.")
            return

        results_dict = st.session_state["results_dict"]
        trained_models = st.session_state["trained_models"]
        y_test = st.session_state["y_test"]
        feature_names = st.session_state["feature_names"]

        # En iyi model
        best_name = max(results_dict, key=lambda k: results_dict[k]["F1-Score"])
        best_f1 = results_dict[best_name]["F1-Score"]

        st.markdown(f"### 🏆 En İyi Model: **{best_name}** (F1-Score: `{best_f1:.4f}`)")

        # Karşılaştırma tablosu
        st.markdown("### 📋 Karşılaştırma Tablosu")
        comparison_data = {
            name: {k: v for k, v in data.items() if k in ["Accuracy", "Precision", "Recall", "F1-Score"]}
            for name, data in results_dict.items()
        }
        comp_df = pd.DataFrame(comparison_data).T.sort_values("F1-Score", ascending=False)
        st.dataframe(
            comp_df.style.format("{:.4f}").highlight_max(axis=0, color="#e94560"),
            use_container_width=True,
        )

        # Metrik karşılaştırma grafiği
        st.markdown("### 📊 Performans Karşılaştırması")
        fig_comp = plot_metric_comparison(results_dict)
        st.pyplot(fig_comp)
        plt.close()

        # Confusion Matrix — en iyi model
        st.markdown(f"### 🔲 Confusion Matrix — {best_name}")
        fig_cm = plot_confusion_matrix(
            y_test, results_dict[best_name]["y_pred"], best_name, best_f1
        )
        st.pyplot(fig_cm)
        plt.close()

        # ROC Curve
        st.markdown("### 📉 ROC Curve Karşılaştırması")
        fig_roc = plot_roc_curves(results_dict, y_test)
        st.pyplot(fig_roc)
        plt.close()

        # Feature Importance
        st.markdown("### 🔍 Özellik Önemleri (Feature Importance)")
        fi_model_name = st.selectbox("Model Seçin:", list(trained_models.keys()))
        fig_fi = plot_feature_importance(
            trained_models[fi_model_name], feature_names, fi_model_name
        )
        if fig_fi:
            st.pyplot(fig_fi)
            plt.close()
        else:
            st.warning("Bu model için feature importance bilgisi mevcut değil.")

        # Özet
        st.markdown("---")
        st.markdown("### 📝 Sonuç Özeti")
        st.success(
            f"**{len(results_dict)}** klasik ML modeli eğitildi ve test seti üzerinde değerlendirildi.\n\n"
            f"🏆 **En yüksek F1-Score:** {best_name} (`{best_f1:.4f}`)\n\n"
            f"Tüm modeller yalnızca hücrenin fiziksel/kimyasal özelliklerini kullanmıştır. "
            f"**Data Leakage önlenmiş**, hiçbir **derin öğrenme** yöntemi kullanılmamıştır."
        )


if __name__ == "__main__":
    main()
