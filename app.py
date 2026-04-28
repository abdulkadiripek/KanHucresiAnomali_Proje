"""
=============================================================================
🩸 Blood Cell Anomaly Detection — Modern Streamlit Dashboard
=============================================================================
Klasik Makine Öğrenmesi algoritmaları ile kan hücresi anomalilerini tespit
eden production-grade interaktif dashboard.

Özellikler:
  - 8 farklı ML algoritması desteği
  - StratifiedKFold cross-validation
  - RandomizedSearchCV hyperparameter tuning
  - SMOTE desteği (imbalanced data)
  - Feature importance & mutual information
  - ROC & Precision-Recall curve
  - Model persistence (kaydet/yükle)
  - Modüler class-based mimari

Derin öğrenme KULLANILMAMISTIR. Sadece klasik ML.

Çalıştırmak için:  streamlit run app.py
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st

from core import DataLoader, Preprocessor, ModelFactory, Evaluator, Visualizer

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
# MODERN CSS — Premium Glassmorphism Dark Theme
# ============================================================================
st.markdown("""
<style>
    /* ========== FONT ========== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    * {
        font-family: 'Inter', sans-serif !important;
    }

    /* ========== ANA ARKA PLAN ========== */
    .stApp {
        background: linear-gradient(160deg, #0a0a1a 0%, #0f0c29 30%, #1a1a2e 60%, #16213e 100%);
    }

    /* ========== SIDEBAR ========== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a 0%, #0f0c29 50%, #12122a 100%);
        border-right: 1px solid rgba(233, 69, 96, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 1.2rem !important;
        letter-spacing: 0.5px;
    }

    /* ========== GLASSMORPHISM KARTLAR ========== */
    div[data-testid="stMetric"] {
        background: rgba(26, 26, 46, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(233, 69, 96, 0.2);
        border-color: rgba(233, 69, 96, 0.3);
    }
    div[data-testid="stMetric"] label {
        color: #a0a0b8 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-weight: 800 !important;
        font-size: 1.8rem !important;
        background: linear-gradient(135deg, #e94560, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ========== BAŞLIKLAR ========== */
    h1 {
        background: linear-gradient(135deg, #e94560 0%, #ff6b6b 40%, #feca57 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        font-size: 2.4rem !important;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem !important;
    }
    h2 {
        color: #e94560 !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        border-bottom: 2px solid rgba(233, 69, 96, 0.2);
        padding-bottom: 10px;
        margin-top: 1.5rem !important;
    }
    h3 {
        color: #feca57 !important;
        font-weight: 600 !important;
        font-size: 1.15rem !important;
    }

    /* ========== TAB ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: rgba(10, 10, 26, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 14px;
        padding: 5px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 24px;
        color: #7a7a9e;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e94560;
        background: rgba(233, 69, 96, 0.08);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #e94560, #c0392b) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);
    }

    /* ========== BUTONLAR ========== */
    .stButton > button {
        background: linear-gradient(135deg, #e94560 0%, #c0392b 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 14px 36px !important;
        font-size: 15px !important;
        letter-spacing: 0.3px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px rgba(233, 69, 96, 0.35) !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 8px 30px rgba(233, 69, 96, 0.55) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) scale(0.98) !important;
    }

    /* ========== EXPANDER ========== */
    .streamlit-expanderHeader {
        background: rgba(26, 26, 46, 0.5) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        font-weight: 600 !important;
    }

    /* ========== DATAFRAME ========== */
    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* ========== SLIDER ========== */
    .stSlider > div > div > div {
        background: #e94560 !important;
    }
    .stSlider > div > div > div > div {
        background: #e94560 !important;
        box-shadow: 0 0 10px rgba(233, 69, 96, 0.5);
    }

    /* ========== DIVIDER ========== */
    hr {
        border-color: rgba(233, 69, 96, 0.12) !important;
        margin: 1.5rem 0 !important;
    }

    /* ========== ALERT ========== */
    .stAlert {
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
    }

    /* ========== MULTISELECT ========== */
    .stMultiSelect [data-baseweb="tag"] {
        background: linear-gradient(135deg, #e94560, #c0392b) !important;
        border-radius: 8px !important;
    }

    /* ========== SELECT BOX ========== */
    .stSelectbox [data-baseweb="select"] > div {
        background: rgba(26, 26, 46, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }

    /* ========== SUCCESS/INFO/WARNING KUTU ========== */
    div[data-testid="stNotification"] {
        border-radius: 12px !important;
    }

    /* ========== PROGRESS BAR ========== */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #e94560, #feca57) !important;
        border-radius: 8px;
    }

    /* ========== CUSTOM BADGE ========== */
    .badge-container {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin: 8px 0;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-red { background: rgba(233, 69, 96, 0.2); color: #e94560; border: 1px solid rgba(233, 69, 96, 0.3); }
    .badge-yellow { background: rgba(254, 202, 87, 0.15); color: #feca57; border: 1px solid rgba(254, 202, 87, 0.3); }
    .badge-blue { background: rgba(0, 210, 255, 0.15); color: #00d2ff; border: 1px solid rgba(0, 210, 255, 0.3); }
    .badge-green { background: rgba(123, 237, 159, 0.15); color: #7bed9f; border: 1px solid rgba(123, 237, 159, 0.3); }

    /* ========== INFO CARD ========== */
    .info-card {
        background: rgba(26, 26, 46, 0.5);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
    }
    .info-card h4 {
        color: #feca57 !important;
        margin-bottom: 12px;
        font-size: 1rem;
    }
    .info-card p {
        color: #a0a0b8;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* ========== STEP INDICATOR ========== */
    .step-item {
        display: flex;
        align-items: flex-start;
        gap: 16px;
        margin: 16px 0;
        padding: 16px 20px;
        background: rgba(26, 26, 46, 0.4);
        border-radius: 14px;
        border-left: 3px solid #e94560;
        transition: all 0.3s ease;
    }
    .step-item:hover {
        background: rgba(233, 69, 96, 0.06);
        transform: translateX(4px);
    }
    .step-number {
        background: linear-gradient(135deg, #e94560, #c0392b);
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.85rem;
        flex-shrink: 0;
    }
    .step-content {
        flex: 1;
    }
    .step-title {
        color: #ffffff;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 4px;
    }
    .step-desc {
        color: #7a7a9e;
        font-size: 0.82rem;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# CONSTANTS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "archive")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

ALL_MODELS = [
    "XGBoost",
    "LightGBM",
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting",
    "AdaBoost",
    "Logistic Regression",
]


# ============================================================================
# ANA UYGULAMA
# ============================================================================
def main():
    # Modüler bileşenler
    loader = DataLoader(data_dir=DATA_DIR)
    viz = Visualizer()

    # ------------------------------------------------------------------
    # SIDEBAR
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown("## 🔬 Kontrol Paneli")
        st.markdown("---")

        # Veri ayarları
        st.markdown("### 📊 Veri Ayarları")
        test_size = st.slider(
            "Test Seti Oranı (%)", 10, 40, 20, 5,
            help="Verinin yüzde kaçı test için ayrılsın?",
        ) / 100.0

        apply_smote = st.checkbox(
            "SMOTE Uygula",
            value=False,
            help="Dengesiz veri için SMOTE over-sampling uygular (sadece train setine).",
        )

        st.markdown("---")

        # Model seçimi
        st.markdown("### 🤖 Model Seçimi")
        selected_models = st.multiselect(
            "Eğitilecek Modeller",
            ALL_MODELS,
            default=["XGBoost", "Random Forest", "LightGBM"],
            help="En az 1 model seçmelisiniz.",
        )

        st.markdown("---")

        # Eğitim modu
        st.markdown("### 🎯 Eğitim Modu")
        training_mode = st.radio(
            "Mod Seçin",
            ["Manuel Parametreler", "Otomatik Tuning (RandomizedSearchCV)"],
            index=0,
            help="Otomatik tuning en iyi hiperparametreleri arar (daha uzun sürer).",
        )

        run_cv = st.checkbox(
            "Cross-Validation (5-Fold)",
            value=True,
            help="StratifiedKFold cross-validation çalıştır.",
        )

        st.markdown("---")

        # Hiperparametre ayarları (sadece manuel modda)
        model_params = {}
        if training_mode == "Manuel Parametreler":
            st.markdown("### ⚙️ Hiperparametreler")

            if "XGBoost" in selected_models:
                with st.expander("🔷 XGBoost", expanded=False):
                    model_params["XGBoost"] = {
                        "n_estimators": st.slider("n_estimators", 50, 1000, 300, 50, key="xgb_n"),
                        "max_depth": st.slider("max_depth", 2, 15, 6, 1, key="xgb_d"),
                        "learning_rate": st.slider("learning_rate", 0.01, 0.5, 0.1, 0.01, key="xgb_lr"),
                        "subsample": st.slider("subsample", 0.5, 1.0, 0.8, 0.05, key="xgb_ss"),
                        "colsample_bytree": st.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.05, key="xgb_cs"),
                    }

            if "LightGBM" in selected_models:
                with st.expander("💡 LightGBM", expanded=False):
                    model_params["LightGBM"] = {
                        "n_estimators": st.slider("n_estimators", 50, 1000, 300, 50, key="lgb_n"),
                        "max_depth": st.slider("max_depth (0=Auto)", 0, 15, 0, 1, key="lgb_d"),
                        "learning_rate": st.slider("learning_rate", 0.01, 0.5, 0.1, 0.01, key="lgb_lr"),
                        "subsample": st.slider("subsample", 0.5, 1.0, 0.8, 0.05, key="lgb_ss"),
                        "colsample_bytree": st.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.05, key="lgb_cs"),
                    }
                    # max_depth=0 → LightGBM'de -1 (sınırsız)
                    if model_params["LightGBM"]["max_depth"] == 0:
                        model_params["LightGBM"]["max_depth"] = -1

            if "Random Forest" in selected_models:
                with st.expander("🌲 Random Forest", expanded=False):
                    model_params["Random Forest"] = {
                        "n_estimators": st.slider("n_estimators", 50, 1000, 300, 50, key="rf_n"),
                        "max_depth": st.slider("max_depth (0=None)", 0, 30, 0, 1, key="rf_d"),
                        "min_samples_split": st.slider("min_samples_split", 2, 20, 5, 1, key="rf_mss"),
                        "min_samples_leaf": st.slider("min_samples_leaf", 1, 10, 2, 1, key="rf_msl"),
                    }
                    if model_params["Random Forest"]["max_depth"] == 0:
                        model_params["Random Forest"]["max_depth"] = None

            if "Extra Trees" in selected_models:
                with st.expander("🌳 Extra Trees", expanded=False):
                    model_params["Extra Trees"] = {
                        "n_estimators": st.slider("n_estimators", 50, 1000, 300, 50, key="et_n"),
                        "max_depth": st.slider("max_depth (0=None)", 0, 30, 0, 1, key="et_d"),
                        "min_samples_split": st.slider("min_samples_split", 2, 20, 5, 1, key="et_mss"),
                        "min_samples_leaf": st.slider("min_samples_leaf", 1, 10, 2, 1, key="et_msl"),
                    }
                    if model_params["Extra Trees"]["max_depth"] == 0:
                        model_params["Extra Trees"]["max_depth"] = None

            if "Gradient Boosting" in selected_models:
                with st.expander("📈 Gradient Boosting", expanded=False):
                    model_params["Gradient Boosting"] = {
                        "n_estimators": st.slider("n_estimators", 50, 500, 200, 50, key="gb_n"),
                        "max_depth": st.slider("max_depth", 2, 10, 5, 1, key="gb_d"),
                        "learning_rate": st.slider("learning_rate", 0.01, 0.5, 0.1, 0.01, key="gb_lr"),
                        "subsample": st.slider("subsample", 0.5, 1.0, 0.8, 0.05, key="gb_ss"),
                    }

            if "AdaBoost" in selected_models:
                with st.expander("⚡ AdaBoost", expanded=False):
                    model_params["AdaBoost"] = {
                        "n_estimators": st.slider("n_estimators", 50, 500, 200, 50, key="ab_n"),
                        "learning_rate": st.slider("learning_rate", 0.01, 2.0, 0.1, 0.01, key="ab_lr"),
                    }

            if "Logistic Regression" in selected_models:
                with st.expander("📐 Logistic Regression", expanded=False):
                    model_params["Logistic Regression"] = {
                        "max_iter": st.slider("max_iter", 100, 5000, 1000, 100, key="lr_mi"),
                    }

            st.markdown("---")

        train_button = st.button("🚀 Modelleri Eğit", type="primary")

        # Footer
        st.markdown("---")
        st.markdown(
            '<div style="text-align:center; color:#5a5a7a; font-size:0.75rem;">'
            'Klasik ML • Derin Öğrenme Yok<br>'
            'Modüler Mimari v2.0'
            '</div>',
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # ANA İÇERİK — HEADER
    # ------------------------------------------------------------------
    st.markdown("# 🩸 Kan Hücresi Anomali Tespiti")

    st.markdown(
        '<div class="badge-container">'
        '<span class="badge badge-red">Klasik ML</span>'
        '<span class="badge badge-yellow">Modüler Mimari</span>'
        '<span class="badge badge-blue">Cross-Validation</span>'
        '<span class="badge badge-green">Production-Grade</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "> **Klasik Makine Öğrenmesi** ile kan hücrelerindeki anomalileri tespit eden "
        "production-grade dashboard. Derin öğrenme **kullanılmamıştır**."
    )

    # Veri yükle
    raw_df = loader.load()
    summary = loader.get_summary(raw_df)

    # ---- TABS ----
    tab_data, tab_eda, tab_train, tab_results = st.tabs([
        "📋 Veri Seti", "📊 Keşifsel Analiz", "🤖 Eğitim", "📈 Sonuçlar"
    ])

    # ==================================================================
    # TAB 1 — VERİ SETİ
    # ==================================================================
    with tab_data:
        st.markdown("## 📋 Veri Seti Genel Bakış")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Toplam Satır", f"{summary['n_rows']:,}")
        c2.metric("Toplam Sütun", f"{summary['n_cols']}")
        c3.metric("Anomali Sayısı", f"{summary['n_anomalies']:,}")
        c4.metric("Anomali Oranı", f"{summary['anomaly_ratio']:.1%}")

        # Ek metrikler
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Eksik Değer", f"{summary['missing_values']}")
        c6.metric("Tekrar Eden Satır", f"{summary['duplicate_rows']}")
        num_count = raw_df.select_dtypes(include=["number"]).shape[1]
        cat_count = raw_df.select_dtypes(include=["object"]).shape[1]
        c7.metric("Sayısal Özellik", f"{num_count}")
        c8.metric("Kategorik Özellik", f"{cat_count}")

        st.markdown("### 🔍 İlk 10 Satır")
        st.dataframe(raw_df.head(10), width="stretch", height=400)

        st.markdown("### 📊 Sütun Detayları")
        col_info = loader.get_column_info(raw_df)
        st.dataframe(col_info, width="stretch", height=400)

        st.markdown("### 🧹 Kaldırılacak Sütunlar")
        st.markdown(
            '<div class="info-card">'
            '<h4>🛡️ Data Leakage Önlemi</h4>'
            '<p>Aşağıdaki sütunlar eğitimden çıkarılır:</p>'
            '<p>'
            '• <strong>cell_id</strong> — Kimlik bilgisi, modele katkısı yok<br>'
            '• <strong>disease_category</strong> — Doğrudan hedef ile ilişkili<br>'
            '• <strong>cell_type</strong> — Doğrudan hedef ile ilişkili<br>'
            '• <strong>cytodiffusion_anomaly_score</strong> — Anomali sızıntısı<br>'
            '• <strong>cytodiffusion_classification_confidence</strong> — Sınıflandırma sızıntısı<br>'
            '• <strong>labeller_confidence_score</strong> — Etiketleyici güven sızıntısı'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### 📈 Temel İstatistikler")
        clean_df_stats = loader.clean(raw_df.copy())
        st.dataframe(
            clean_df_stats.describe().T.style.format("{:.3f}"),
            width="stretch",
            height=400,
        )

    # ==================================================================
    # TAB 2 — EDA
    # ==================================================================
    with tab_eda:
        st.markdown("## 📊 Keşifsel Veri Analizi")

        df_clean = loader.clean(raw_df.copy())

        # Anomali dağılımı
        st.markdown("### 🎯 Anomali Dağılımı")
        col_a, col_b = st.columns([1, 2])

        with col_a:
            fig_pie = viz.plot_class_distribution(raw_df["anomaly_label"])
            st.pyplot(fig_pie)
            plt.close()

        with col_b:
            fig_ct = viz.plot_cell_type_distribution(raw_df)
            st.pyplot(fig_ct)
            plt.close()

        # Box plot karşılaştırması
        st.markdown("### 📦 Normal vs Anomali Karşılaştırması")
        num_cols = df_clean.select_dtypes(include=["number"]).columns.drop("anomaly_label").tolist()

        default_box_feats = num_cols[:6] if len(num_cols) >= 6 else num_cols
        selected_box_feats = st.multiselect(
            "Özellik Seçin (Box Plot):", num_cols,
            default=default_box_feats, key="box_feats",
        )
        if selected_box_feats:
            fig_box = viz.plot_box_anomaly_comparison(df_clean, selected_box_feats)
            st.pyplot(fig_box)
            plt.close()

        # Histogram
        st.markdown("### 📈 Sayısal Özellik Dağılımları")
        selected_feat = st.selectbox("Özellik Seçin:", num_cols, index=0)
        fig_hist = viz.plot_feature_histogram(df_clean, selected_feat)
        st.pyplot(fig_hist)
        plt.close()

        # Korelasyon matrisi
        st.markdown("### 🔗 Korelasyon Matrisi")
        fig_corr = viz.plot_correlation_matrix(df_clean)
        st.pyplot(fig_corr)
        plt.close()

    # ==================================================================
    # TAB 3 — EĞİTİM
    # ==================================================================
    with tab_train:
        st.markdown("## 🤖 Model Eğitimi")

        if not selected_models:
            st.error("⚠️ Lütfen sol panelden en az 1 model seçin.")
            return

        if not train_button:
            st.info(
                "👈 Sol paneldeki kontrolleri ayarlayın ve **'🚀 Modelleri Eğit'** "
                "butonuna tıklayın."
            )

            # Pipeline adımları
            st.markdown("### 📝 Eğitim Pipeline Adımları")
            steps = [
                ("Veri Temizleme", "Leakage ve kimlik sütunları kaldırılır"),
                ("One-Hot Encoding", "Kategorik değişkenler sayısallaştırılır"),
                ("Train/Test Split", "Stratified bölünme (sınıf dengesi korunur)"),
                ("StandardScaler", "Sayısal özellikler ölçeklenir (fit → train)"),
                ("SMOTE (Opsiyonel)", "Dengesiz veri için over-sampling"),
                ("Cross-Validation", "5-Fold StratifiedKFold ile güvenilirlik testi"),
                ("Model Eğitimi", "Seçilen algoritmalar eğitilir"),
                ("Değerlendirme", "12+ metrik üretilir (MCC, Kappa, PR-AUC vb.)"),
            ]
            steps_html = ""
            for i, (title, desc) in enumerate(steps, 1):
                steps_html += (
                    f'<div class="step-item">'
                    f'<div class="step-number">{i}</div>'
                    f'<div class="step-content">'
                    f'<div class="step-title">{title}</div>'
                    f'<div class="step-desc">{desc}</div>'
                    f'</div></div>'
                )
            st.markdown(steps_html, unsafe_allow_html=True)

            st.markdown("### 🤖 Desteklenen Modeller")
            st.markdown(
                '<div class="badge-container">'
                '<span class="badge badge-red">XGBoost</span>'
                '<span class="badge badge-yellow">LightGBM</span>'
                '<span class="badge badge-green">Random Forest</span>'
                '<span class="badge badge-blue">Extra Trees</span>'
                '<span class="badge badge-red">Gradient Boosting</span>'
                '<span class="badge badge-yellow">AdaBoost</span>'
                '<span class="badge badge-blue">Logistic Regression</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            return

        # ---- EĞİTİM BAŞLAT ----
        progress = st.progress(0, text="Veri hazırlanıyor...")

        # Veri temizle
        df_clean = loader.clean(raw_df.copy())
        progress.progress(5, text="Veri temizlendi. Ön işleme yapılıyor...")

        # Preprocessor
        preprocessor = Preprocessor(
            test_size=test_size,
            apply_smote=apply_smote,
        )
        split = preprocessor.process(df_clean)
        progress.progress(15, text="Ön işleme tamamlandı.")

        # Bilgi
        smote_text = " (SMOTE uygulandı)" if apply_smote else ""
        st.success(
            f"✅ Veri hazır — Train: **{split.X_train.shape[0]:,}** satır{smote_text}, "
            f"Test: **{split.X_test.shape[0]:,}** satır, "
            f"Özellik: **{split.X_train.shape[1]}**"
        )

        # Model eğitimi
        factory = ModelFactory()
        evaluator = Evaluator()
        train_results = {}  # name -> TrainResult
        is_tuning = training_mode == "Otomatik Tuning (RandomizedSearchCV)"

        for i, name in enumerate(selected_models):
            pct = 15 + int(((i + 1) / len(selected_models)) * 75)
            if is_tuning:
                progress.progress(min(pct, 90), text=f"{name} — Hyperparameter tuning...")
                try:
                    result = factory.tune_hyperparameters(
                        name, split.X_train, split.y_train,
                        split.X_test, n_iter=20,
                    )
                except ValueError:
                    # Param grid tanımlı değilse normal eğit
                    model = factory.create_model(name, model_params.get(name, {}))
                    result = factory.train(
                        name, model, split.X_train, split.y_train,
                        split.X_test, split.y_test, run_cv=run_cv,
                    )
            else:
                progress.progress(min(pct, 90), text=f"{name} eğitiliyor...")
                params = model_params.get(name, {})
                model = factory.create_model(name, params)
                result = factory.train(
                    name, model, split.X_train, split.y_train,
                    split.X_test, split.y_test, run_cv=run_cv,
                )

            train_results[result.model_name] = result

            # Evaluate
            evaluator.evaluate(
                model_name=result.model_name,
                y_true=split.y_test,
                y_pred=result.y_pred,
                y_proba=result.y_proba,
                cv_mean=result.cv_mean,
                cv_std=result.cv_std,
            )

        progress.progress(100, text="✅ Tüm modeller eğitildi!")

        # Session state'e kaydet
        st.session_state["train_results"] = train_results
        st.session_state["evaluator"] = evaluator
        st.session_state["y_test"] = split.y_test
        st.session_state["feature_names"] = split.feature_names
        st.session_state["trained"] = True

        # Hızlı özet
        st.markdown("### 📊 Eğitim Özeti")
        for name, result in train_results.items():
            eval_res = evaluator.results[name]
            with st.expander(f"✅ {name}", expanded=True):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{eval_res.accuracy:.4f}")
                m2.metric("Precision", f"{eval_res.precision:.4f}")
                m3.metric("Recall", f"{eval_res.recall:.4f}")
                m4.metric("F1-Score", f"{eval_res.f1:.4f}")

                m5, m6, m7, m8 = st.columns(4)
                m5.metric("MCC", f"{eval_res.mcc:.4f}")
                m6.metric("Kappa", f"{eval_res.kappa:.4f}")
                if eval_res.roc_auc is not None:
                    m7.metric("ROC-AUC", f"{eval_res.roc_auc:.4f}")
                if eval_res.cv_mean is not None:
                    m8.metric("CV Mean (F1)", f"{eval_res.cv_mean:.4f} ± {eval_res.cv_std:.4f}" if eval_res.cv_std else f"{eval_res.cv_mean:.4f}")

                if result.best_params:
                    st.markdown("**🔧 En İyi Parametreler:**")
                    st.json(result.best_params)

    # ==================================================================
    # TAB 4 — SONUÇLAR
    # ==================================================================
    with tab_results:
        st.markdown("## 📈 Detaylı Sonuçlar ve Karşılaştırma")

        if "trained" not in st.session_state or not st.session_state["trained"]:
            st.info("⏳ Henüz model eğitilmedi. 'Eğitim' sekmesinden modelleri eğitin.")
            return

        train_results = st.session_state["train_results"]
        evaluator = st.session_state["evaluator"]
        y_test = st.session_state["y_test"]
        feature_names = st.session_state["feature_names"]

        # En iyi model
        best_name = evaluator.get_best_model("F1-Score")
        best_eval = evaluator.results[best_name]

        st.markdown(
            f'<div class="info-card">'
            f'<h4>🏆 En İyi Model: {best_name}</h4>'
            f'<p>'
            f'F1-Score: <strong>{best_eval.f1:.4f}</strong> · '
            f'Accuracy: <strong>{best_eval.accuracy:.4f}</strong> · '
            f'MCC: <strong>{best_eval.mcc:.4f}</strong>'
            f'</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Karşılaştırma tablosu
        st.markdown("### 📋 Karşılaştırma Tablosu")
        comp_df = evaluator.get_comparison_table()
        st.dataframe(
            comp_df.style.format("{:.4f}").highlight_max(axis=0, color="#e94560"),
            width="stretch",
        )

        # Metrik karşılaştırma grafiği
        st.markdown("### 📊 Performans Karşılaştırması")
        fig_comp = viz.plot_metric_comparison(comp_df)
        if fig_comp:
            st.pyplot(fig_comp)
            plt.close()

        # Cross-validation dağılımı
        cv_data = {}
        for name, res in train_results.items():
            if res.cv_scores is not None:
                cv_data[name] = res.cv_scores
        if cv_data:
            st.markdown("### 📉 Cross-Validation Skor Dağılımı")
            fig_cv = viz.plot_cv_scores(cv_data)
            st.pyplot(fig_cv)
            plt.close()

        # Confusion Matrix
        st.markdown(f"### 🔲 Confusion Matrix — {best_name}")
        fig_cm = viz.plot_confusion_matrix(
            y_test, train_results[best_name].y_pred, best_name, best_eval.f1,
        )
        st.pyplot(fig_cm)
        plt.close()

        # ROC Curve
        st.markdown("### 📉 ROC Curve Karşılaştırması")
        roc_data = {}
        for name, res in train_results.items():
            if res.y_proba is not None:
                fpr, tpr, roc_auc_val = evaluator.get_roc_data(y_test, res.y_proba)
                roc_data[name] = (fpr, tpr, roc_auc_val)
        if roc_data:
            fig_roc = viz.plot_roc_curves(roc_data)
            st.pyplot(fig_roc)
            plt.close()

        # Precision-Recall Curve
        st.markdown("### 📈 Precision-Recall Curve")
        pr_data = {}
        for name, res in train_results.items():
            if res.y_proba is not None:
                prec_arr, rec_arr, pr_auc_val = evaluator.get_pr_data(y_test, res.y_proba)
                pr_data[name] = (prec_arr, rec_arr, pr_auc_val)
        if pr_data:
            fig_pr = viz.plot_pr_curves(pr_data)
            st.pyplot(fig_pr)
            plt.close()

        # Feature Importance
        st.markdown("### 🔍 Özellik Önemleri")
        fi_model_name = st.selectbox("Model Seçin:", list(train_results.keys()), key="fi_select")
        fig_fi = viz.plot_feature_importance(
            train_results[fi_model_name].model, feature_names, fi_model_name,
        )
        if fig_fi:
            st.pyplot(fig_fi)
            plt.close()
        else:
            st.warning("Bu model için feature importance bilgisi mevcut değil.")

        # Classification Report
        st.markdown("### 📝 Sınıflandırma Raporu")
        cr_model = st.selectbox("Model Seçin:", list(evaluator.results.keys()), key="cr_select")
        st.code(evaluator.results[cr_model].classification_rep, language="text")

        # Model Kaydetme
        st.markdown("---")
        st.markdown("### 💾 Model Kaydet")
        save_col1, save_col2 = st.columns([2, 1])
        with save_col1:
            save_model_name = st.selectbox(
                "Kaydedilecek Model:", list(train_results.keys()), key="save_select",
            )
        with save_col2:
            if st.button("💾 Kaydet", key="save_btn"):
                safe_name = save_model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                filepath = os.path.join(MODELS_DIR, f"{safe_name}.joblib")
                saved = ModelFactory.save_model(train_results[save_model_name].model, filepath)
                st.success(f"✅ Model kaydedildi: `{saved}`")

        # Sonuç özeti
        st.markdown("---")
        st.markdown(
            f'<div class="info-card">'
            f'<h4>📝 Sonuç Özeti</h4>'
            f'<p>'
            f'<strong>{len(train_results)}</strong> klasik ML modeli eğitildi ve '
            f'test seti üzerinde değerlendirildi.<br><br>'
            f'🏆 <strong>En yüksek F1-Score:</strong> {best_name} '
            f'(<strong>{best_eval.f1:.4f}</strong>)<br><br>'
            f'Tüm modeller yalnızca hücrenin fiziksel/kimyasal özelliklerini kullanmıştır. '
            f'Data Leakage önlenmiş, hiçbir derin öğrenme yöntemi kullanılmamıştır.'
            f'</p>'
            f'</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
