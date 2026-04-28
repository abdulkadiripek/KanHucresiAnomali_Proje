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
    page_title="Kan Hücresi Anomali Tespiti",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# MODERN CSS — Premium Glassmorphism Dark Theme
# ============================================================================
st.markdown("""
<style>
    /* ========== PREMIUM GLASS & NEON DASHBOARD THEME ========== */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Deep premium background */
    .stApp {
        background: linear-gradient(145deg, #0f172a 0%, #020617 100%);
    }
    
    .block-container {
        padding-top: 2rem !important;
        max-width: 1300px; 
    }

    /* Vibrant Crisp Headers */
    h1 {
        font-weight: 700 !important;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem !important;
        letter-spacing: -0.5px;
    }
    
    h2 {
        color: #f8fafc !important;
        font-weight: 600 !important;
        border-bottom: 2px solid rgba(56, 189, 248, 0.2);
        padding-bottom: 8px;
    }

    h3 {
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }
    
    /* Sexy Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(56, 189, 248, 0.15);
        border-color: rgba(56, 189, 248, 0.3);
    }
    div[data-testid="stMetricValue"] {
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        background: linear-gradient(to right, #38bdf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.8rem !important;
    }
    
    /* Glowing Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.5px;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
    }
    .stButton > button[data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.6) !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Input widgets styling */
    .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div {
        background-color: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    
    /* Tabs minimalism */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        color: #94a3b8;
        font-weight: 500;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #38bdf8, #3b82f6) !important;
        color: white !important;
        box-shadow: 0 2px 10px rgba(56, 189, 248, 0.3);
    }
    
    /* Dataframes */
    .stDataFrame {
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        overflow: hidden;
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
        st.markdown("## 🎛️ Kontrol Paneli")
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
        st.caption("Klasik ML • Derin Öğrenme Yok • Modüler Mimari v2.0")

    # ------------------------------------------------------------------
    # ANA İÇERİK — HEADER
    # ------------------------------------------------------------------
    st.markdown("## 🩸 Kan Hücresi Anomali Tespiti")
    st.caption("✨ Klasik ML · Modüler Mimari · Cross-Validation · Production-Grade")

    st.markdown(
        "Klasik Makine Öğrenmesi ile kan hücrelerindeki anomalileri tespit eden "
        "production-grade dashboard. Derin öğrenme kullanılmamıştır."
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
        st.markdown("### 📋 Veri Seti Genel Bakış")

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
        st.warning(
            "**🛡️ Data Leakage Önlemi** — Eğitimden çıkarılan sütunlar:\n"
            "* **cell_id, cell_type, disease_category**: Doğrudan hedef ile ilişkili veya kimlik.\n"
            "* **cytodiffusion_anomaly_score, cytodiffusion_classification_confidence, labeller_confidence_score**: Anomali sızıntıları."
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
        st.markdown("### 📊 Keşifsel Veri Analizi")
        st.caption(
            "Her grafik bir tasarım kararını kanıtlar. Özellikler kullanıcı tarafından "
            "değil, **istatistiksel kriterlere göre veri tarafından** seçilmiştir."
        )

        df_clean = loader.clean(raw_df.copy())

        # ----- GRAFİK 1: Leakage Kanıtı -----
        st.markdown("#### 1️⃣ Veri Sızıntısı (Data Leakage) Kanıtı")
        st.markdown(
            "**Soru:** Hangi sütunları neden modelden çıkardık?  \n"
            "**Yöntem:** Tüm sayısal sütunların `anomaly_label` ile mutlak Pearson korelasyonu."
        )
        leakage_all = loader.leakage_cols + loader.id_cols
        fig_leak = viz.plot_leakage_evidence(
            raw_df, target_col="anomaly_label", leakage_cols=leakage_all,
        )
        st.pyplot(fig_leak)
        plt.close()
        st.info(
            "**Yorum:** `cytodiffusion_anomaly_score` ve `labeller_confidence_score` "
            "hedef değişkenle çok yüksek korelasyona sahip — bunlar başka bir modelin/uzmanın "
            "etiketleme sürecinden gelen bilgilerdir. Modele dahil edilseydi gerçek genelleme "
            "değil **leakage** ölçerdik. Bu yüzden eğitime alınmadılar."
        )

        st.markdown("---")

        # ----- GRAFİK 2: Sınıf Dengesizliği -----
        st.markdown("#### 2️⃣ Sınıf Dengesizliği")
        st.markdown(
            "**Soru:** Veri ne kadar dengesiz? Bu, tasarım seçimlerini nasıl etkiliyor?"
        )
        col_donut, col_text = st.columns([1.2, 1])
        with col_donut:
            fig_donut = viz.plot_class_imbalance_donut(raw_df["anomaly_label"])
            st.pyplot(fig_donut)
            plt.close()
        with col_text:
            anomaly_pct = raw_df["anomaly_label"].mean() * 100
            st.markdown(
                f"**Anomali oranı: %{anomaly_pct:.1f}**  \n\n"
                "**Tasarım kararları:**  \n"
                "🔹 **`stratify=y`** — Train/test split sınıf oranını koruyacak şekilde yapılır.  \n"
                "🔹 **F1-Score** birincil metrik — Accuracy dengesiz veride yanıltıcıdır.  \n"
                "🔹 **SMOTE opsiyonel** — Azınlık sınıfı için sentetik örnekleme (sidebar'dan açılabilir).  \n"
                "🔹 **Class-balanced metrikler** — Balanced Accuracy, MCC, Cohen's Kappa raporlanır."
            )

        st.markdown("---")

        # ----- GRAFİK 3: Cohen's d ile Top Özellikler -----
        st.markdown("#### 3️⃣ En Ayırt Edici Özellikler — Cohen's d Etki Büyüklüğü")
        st.markdown(
            "**Soru:** Hangi özellikler Normal ile Anomali arasında en güçlü farkı yaratıyor?  \n"
            "**Yöntem:** Her sayısal özellik için iki sınıfın ortalamaları arasındaki "
            "**standardize edilmiş farkı** (Cohen's d) hesapladık. "
            "Eşikler: 0.2 küçük · 0.5 orta · 0.8 büyük etki."
        )
        fig_cohen, top_features = viz.plot_cohens_d_top_features(
            df_clean, target_col="anomaly_label", top_n=10,
        )
        st.pyplot(fig_cohen)
        plt.close()
        if top_features:
            top_3_str = ", ".join(f"`{f}`" for f in top_features[:3])
            st.info(
                f"**Yorum:** En güçlü ayrım sinyalleri {top_3_str} özelliklerinden geliyor. "
                "Bu liste subjektif değil — tüm sayısal özellikler tarandı, etki büyüklüğüne göre "
                "otomatik sıralandı. Modelin hangi sinyallerden öğrendiğini istatistiksel olarak öngörüyoruz."
            )

        st.markdown("---")

        # ----- GRAFİK 4: Top 4 Özellik için KDE Dağılımı -----
        st.markdown("#### 4️⃣ Top 4 Özelliğin Sınıf Bazlı Yoğunluk Dağılımı")
        st.markdown(
            "**Soru:** Cohen's d'ye göre seçilen top özelliklerde sınıflar gerçekten ayrışıyor mu?  \n"
            "**Yöntem:** Yukarıdaki listenin **ilk 4 özelliği** için KDE (yoğunluk eğrisi). "
            "İki dağılım ne kadar az örtüşüyorsa, model o özellikten o kadar bilgi çıkarabilir."
        )
        top_4 = top_features[:4] if top_features else []
        if top_4:
            fig_kde = viz.plot_top_features_kde(
                df_clean, features=top_4, target_col="anomaly_label",
            )
            if fig_kde is not None:
                st.pyplot(fig_kde)
                plt.close()
            st.success(
                "**Yorum:** Mavi (Normal) ve kırmızı (Anomali) yoğunlukları belirgin biçimde "
                "kayık → klasik ML algoritmaları (XGBoost, RF, LightGBM) bu özelliklerdeki "
                "eşikleri öğrenerek başarılı sınıflandırma yapabilir. Tam örtüşme olsaydı "
                "model bu özellikten faydalanamazdı."
            )

        st.markdown("---")

        # ----- GRAFİK 5: Kategorik Bias Kontrolü -----
        st.markdown("#### 5️⃣ Kategorik Özelliklerde Anomali Oranı — Bias Kontrolü")
        st.markdown(
            "**Soru:** Anomaliler bazı klinik gruplara veya teknik koşullara göre yoğunlaşıyor mu?  \n"
            "**Yöntem:** Her kategorik sütun için kategori bazında anomali oranını "
            "**genel baseline** ile karşılaştırdık. Sapma > %5 ise renkli işaretlendi."
        )
        # Kategorik sütunlar — leakage olanlar zaten df_clean'de yok
        cat_candidates = [
            "patient_age_group", "patient_sex",
            "dataset_source", "staining_protocol",
            "microscope_model", "magnification_x", "image_resolution_px",
        ]
        cat_cols_present = [c for c in cat_candidates if c in df_clean.columns]
        if cat_cols_present:
            # Bias özet tablosu
            bias_df = viz.compute_categorical_bias_summary(
                df_clean, cat_cols_present, target_col="anomaly_label",
            )
            st.markdown("**📋 Bias Özeti (yelpaze = max kategori − min kategori anomali oranı)**")
            st.dataframe(
                bias_df.style.format({"Yelpaze (max-min)": "{:.1%}"}),
                width="stretch", hide_index=True,
            )

            # Grafik
            fig_bias = viz.plot_categorical_bias(
                df_clean, cat_cols_present, target_col="anomaly_label",
            )
            if fig_bias is not None:
                st.pyplot(fig_bias)
                plt.close()

            # Otomatik yorum: en yüksek yelpazeli sütun
            top_bias = bias_df.iloc[0]
            top_col = top_bias["Sütun"]
            top_spread = top_bias["Yelpaze (max-min)"]

            if top_spread < 0.05:
                interpretation = (
                    f"**Yorum:** Hiçbir kategorik sütunda anomali oranı belirgin sapma "
                    f"göstermiyor (en yüksek yelpaze: `{top_col}`, %{top_spread:.1%}). "
                    "→ **İyi haber:** Klinik veya teknik bias yok, model genelleyebilir."
                )
            else:
                interpretation = (
                    f"**Yorum:** En yüksek sapma `{top_col}` sütununda "
                    f"(%{top_spread:.1%} yelpaze) — yani bu kategorik sütunun "
                    f"alt grupları arasında anomali oranı belirgin biçimde değişiyor. "
                )
                # Klinik mi teknik mi yorumla
                if top_col in ("patient_age_group", "patient_sex"):
                    interpretation += (
                        "Bu **klinik anlamlı** bir sinyal olabilir (örn. yaşlılarda hematolojik "
                        "hastalık daha yaygındır). Modelin yakaladığı sinyal tıbbi gerçeklikle uyumlu."
                    )
                else:
                    interpretation += (
                        "Bu **teknik bir bias** olabilir (örn. bir veri kaynağında anomali "
                        "örneklerinin daha fazla toplanmış olması — batch effect). "
                        "Model bu sütunu sinyal olarak kullanabilir, dikkat edilmeli."
                    )
            st.info(interpretation)

    # ==================================================================
    # TAB 3 — EĞİTİM
    # ==================================================================
    with tab_train:
        st.markdown("### 🤖 Model Eğitimi")

        if not selected_models:
            st.error("Lütfen sol panelden en az 1 model seçin.")
            return

        if not train_button:
            st.info(
                "Sol paneldeki kontrolleri ayarlayın ve **'Modelleri Eğit'** "
                "butonuna tıklayın."
            )

            # Pipeline adımları
            st.markdown("#### 📝 Eğitim Pipeline Adımları")
            st.markdown(
                "1. **Veri Temizleme**: Leakage ve kimlik sütunları kaldırılır.\n"
                "2. **One-Hot Encoding**: Kategorik değişkenler sayısallaştırılır.\n"
                "3. **Train/Test Split**: Stratified bölünme ile dengeli dağılım sağlanır.\n"
                "4. **StandardScaler**: Sayısal sütunlar ölçeklenir.\n"
                "5. **SMOTE**: Dengesiz sınıflar üzerinde sentetik veri üretilir (opsiyonel).\n"
                "6. **Cross-Validation**: Modeller K-Fold ile test edilerek çapraz doğrulanır.\n"
                "7. **Değerlendirme**: Eğitim sonrasında zengin metrikler üretilir."
            )

            st.markdown("#### 🤖 Desteklenen Modeller")
            st.caption("XGBoost · LightGBM · Random Forest · Extra Trees · Gradient Boosting · AdaBoost · Logistic Regression")
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
        smote_text = " (SMOTE kullanıldı)" if apply_smote else ""
        st.success(
            f"Veri hazır — Train: **{split.X_train.shape[0]:,}** satır{smote_text}, "
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

        progress.progress(100, text="Tüm modeller eğitildi.")

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
                    m8.metric("CV Mean", f"{eval_res.cv_mean:.4f} ± {eval_res.cv_std:.4f}" if eval_res.cv_std else f"{eval_res.cv_mean:.4f}")

                if result.best_params:
                    st.markdown("**En İyi Parametreler:**")
                    st.json(result.best_params)

    # ==================================================================
    # TAB 4 — SONUÇLAR
    # ==================================================================
    with tab_results:
        st.markdown("### 📈 Detaylı Sonuçlar ve Karşılaştırma")

        if "trained" not in st.session_state or not st.session_state["trained"]:
            st.info("Henüz model eğitilmedi. 'Eğitim' sekmesinden modelleri eğitin.")
            return

        train_results = st.session_state["train_results"]
        evaluator = st.session_state["evaluator"]
        y_test = st.session_state["y_test"]
        feature_names = st.session_state["feature_names"]

        # En iyi model
        best_name = evaluator.get_best_model("F1-Score")
        best_eval = evaluator.results[best_name]

        st.success(
            f"**🏆 En İyi Model: {best_name}**  \n"
            f"F1-Score: `{best_eval.f1:.4f}` · Accuracy: `{best_eval.accuracy:.4f}` · MCC: `{best_eval.mcc:.4f}`"
        )

        # Karşılaştırma tablosu
        st.markdown("#### 📋 Karşılaştırma Tablosu")
        comp_df = evaluator.get_comparison_table()
        st.dataframe(
            comp_df.style.format("{:.4f}").highlight_max(axis=0, color="#e94560"),
            width="stretch",
        )

        # Metrik karşılaştırma grafiği
        st.markdown("#### 📊 Performans Karşılaştırması")
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
            st.markdown("#### 📉 Cross-Validation Skor Dağılımı")
            fig_cv = viz.plot_cv_scores(cv_data)
            st.pyplot(fig_cv)
            plt.close()

        # Confusion Matrix
        st.markdown(f"#### 🔲 Confusion Matrix — {best_name}")
        fig_cm = viz.plot_confusion_matrix(
            y_test, train_results[best_name].y_pred, best_name, best_eval.f1,
        )
        st.pyplot(fig_cm)
        plt.close()

        # ROC Curve
        st.markdown("#### 📉 ROC Curve")
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
        st.markdown("#### 📈 Precision-Recall Curve")
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
        st.markdown("#### 🔍 Özellik Önemleri")
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
        st.markdown("#### 📝 Sınıflandırma Raporu")
        cr_model = st.selectbox("Model Seçin:", list(evaluator.results.keys()), key="cr_select")
        st.code(evaluator.results[cr_model].classification_rep, language="text")

        # Model Kaydetme
        st.markdown("---")
        st.markdown("#### 💾 Model Kaydet")
        save_col1, save_col2 = st.columns([2, 1])
        with save_col1:
            save_model_name = st.selectbox(
                "Kaydedilecek Modeli Seçin:", list(train_results.keys()), key="save_select",
            )
        with save_col2:
            if st.button("Kaydet", key="save_btn"):
                safe_name = save_model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                filepath = os.path.join(MODELS_DIR, f"{safe_name}.joblib")
                saved = ModelFactory.save_model(train_results[save_model_name].model, filepath)
                st.success(f"Model kaydedildi: `{saved}`")

        # Sonuç özeti
        st.markdown("---")
        st.info(
            f"**📝 Sonuç Özeti:**  \n"
            f"Eğitilen modeller içerisinde en yüksek F1-Skoru **{best_name}** tarafından alınıştır "
            f"(F1: `{best_eval.f1:.4f}`). Kaynak verinin özellikleri işlenmiş, sızıntı (data leakage) engellenmiştir."
        )


if __name__ == "__main__":
    main()
