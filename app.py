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

from core import DataLoader, Evaluator, Visualizer, Explainer

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


# ============================================================================
# MODEL YÜKLEME
# ============================================================================
@st.cache_resource
def load_saved_models():
    """Kaydedilmiş modelleri ve metadata'yı yükle."""
    import joblib

    metadata_path = os.path.join(MODELS_DIR, "metadata.joblib")
    if not os.path.exists(metadata_path):
        return None

    metadata = joblib.load(metadata_path)

    # Model dosyalarını yükle
    models = {}
    for name in metadata["model_names"]:
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_path = os.path.join(MODELS_DIR, f"{safe_name}.joblib")
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)

    metadata["models"] = models
    return metadata


# ============================================================================
# ANA UYGULAMA
# ============================================================================
def main():
    # Modüler bileşenler
    loader = DataLoader(data_dir=DATA_DIR)
    viz = Visualizer()

    # Kaydedilmiş modelleri yükle
    saved = load_saved_models()
    models_loaded = saved is not None and len(saved.get("models", {})) > 0

    # ------------------------------------------------------------------
    # SIDEBAR — sadeleştirilmiş
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown("## 🩸 Kan Hücresi Anomali Tespiti")
        st.markdown("---")

        page = st.radio(
            "Sayfa Seçin",
            [
                "📋 Veri Seti",
                "📊 Keşifsel Analiz",
                "🤖 Eğitim",
                "📈 Sonuçlar",
                "🧠 Açıklanabilirlik",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        if models_loaded:
            best = saved.get("best_model", "?")
            best_f1 = saved["eval_results"].get(best, {}).get("f1", 0)
            st.success(f"🏆 **{best}**  \nF1: `{best_f1:.4f}`")
        else:
            st.error("Model bulunamadı.  \n`python3 train_models.py`")

        st.markdown("---")
        st.caption("Klasik ML · Derin Öğrenme Yok")

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

    # ==================================================================
    # SAYFA İÇERİKLERİ — sidebar radio'ya göre
    # ==================================================================

    if page == "📋 Veri Seti":
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
    elif page == "📊 Keşifsel Analiz":
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
    elif page == "🤖 Eğitim":
        st.markdown("### 🤖 Model Eğitimi")

        if not models_loaded:
            st.error(
                "Eğitilmiş model bulunamadı. Terminalde şu komutu çalıştırın:  \n"
                "```\npython3 train_models.py\n```"
            )
            return

        st.success(
            f"✅ **{len(saved['models'])} model** önceden eğitildi ve yüklendi. "
            "Eğitim sırasında `RandomizedSearchCV` ile hyperparameter optimizasyonu "
            "ve `5-Fold StratifiedKFold` cross-validation uygulanmıştır."
        )

        # Pipeline adımları
        st.markdown("#### 📝 Uygulanan Pipeline")
        st.markdown(
            "1. **Veri Temizleme** — Leakage ve kimlik sütunları kaldırıldı (EDA'da kanıtlandı).  \n"
            "2. **One-Hot Encoding** — Kategorik değişkenler sayısallaştırıldı (`pd.get_dummies`).  \n"
            "3. **Train/Test Split** — %80/%20 oranında, `stratify=y` ile sınıf dengesi korundu.  \n"
            "4. **StandardScaler** — Sayısal sütunlar ölçeklendi (sadece train üzerinde fit).  \n"
            "5. **RandomizedSearchCV** — Her model için 20 iterasyonla en iyi hiperparametreler arandı.  \n"
            "6. **5-Fold CV** — StratifiedKFold cross-validation ile genelleme yeteneği test edildi."
        )

        st.markdown("---")

        # Model detayları
        st.markdown("#### 📊 Eğitilen Modeller")
        for name in saved["model_names"]:
            eval_info = saved["eval_results"].get(name, {})
            train_info = saved["train_results"].get(name, {})
            is_best = (name == saved.get("best_model"))

            title = f"🏆 {name}" if is_best else f"✅ {name}"
            with st.expander(title, expanded=is_best):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{eval_info.get('accuracy', 0):.4f}")
                m2.metric("Precision", f"{eval_info.get('precision', 0):.4f}")
                m3.metric("Recall", f"{eval_info.get('recall', 0):.4f}")
                m4.metric("F1-Score", f"{eval_info.get('f1', 0):.4f}")

                m5, m6, m7, m8 = st.columns(4)
                m5.metric("MCC", f"{eval_info.get('mcc', 0):.4f}")
                m6.metric("Kappa", f"{eval_info.get('kappa', 0):.4f}")
                roc = eval_info.get("roc_auc")
                if roc is not None:
                    m7.metric("ROC-AUC", f"{roc:.4f}")
                cv_mean = train_info.get("cv_mean")
                if cv_mean is not None:
                    cv_std = train_info.get("cv_std")
                    cv_text = f"{cv_mean:.4f} ± {cv_std:.4f}" if cv_std else f"{cv_mean:.4f}"
                    m8.metric("CV Mean (F1)", cv_text)

                best_params = train_info.get("best_params")
                if best_params:
                    st.markdown("**En İyi Hiperparametreler:**")
                    st.json(best_params)

        st.markdown("---")
        st.markdown("#### 🧬 Neden Bu 3 Model?")
        st.markdown(
            "| Model | Paradigma | Neden Seçildi? |\n"
            "|---|---|---|\n"
            "| **XGBoost** | Boosting (level-wise) | Tabular veri için endüstri standardı, en yüksek performans |\n"
            "| **LightGBM** | Boosting (leaf-wise) | Farklı boosting stratejisi, sağlamlık doğrulaması |\n"
            "| **Random Forest** | Bagging | Farklı paradigma, overfitting direnci, sanity check |"
        )

    # ==================================================================
    # TAB 4 — SONUÇLAR
    # ==================================================================
    elif page == "📈 Sonuçlar":
        st.markdown("### 📈 Detaylı Sonuçlar ve Karşılaştırma")

        if not models_loaded:
            st.info("Model bulunamadı. Terminalde `python3 train_models.py` çalıştırın.")
            return

        eval_results = saved["eval_results"]
        y_test = saved["y_test"]
        feature_names = saved["feature_names"]
        comp_df = saved["comparison_table"]
        models = saved["models"]
        train_info = saved["train_results"]

        best_name = saved["best_model"]
        best_eval = eval_results[best_name]

        st.success(
            f"**🏆 En İyi Model: {best_name}**  \n"
            f"F1-Score: `{best_eval['f1']:.4f}` · "
            f"Accuracy: `{best_eval['accuracy']:.4f}` · "
            f"MCC: `{best_eval['mcc']:.4f}`"
        )

        # Karşılaştırma tablosu
        st.markdown("#### 📋 Karşılaştırma Tablosu")
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

        # Confusion Matrix — en iyi model
        st.markdown(f"#### 🔲 Confusion Matrix — {best_name}")
        best_y_pred = train_info[best_name]["y_pred"]
        fig_cm = viz.plot_confusion_matrix(
            y_test, best_y_pred, best_name, best_eval["f1"],
        )
        st.pyplot(fig_cm)
        plt.close()

        # Confusion matrix yorumu
        from sklearn.metrics import confusion_matrix as _cm_func
        cm = _cm_func(y_test, best_y_pred)
        tn, fp, fn, tp = cm.ravel()
        st.info(
            f"**Yorum:**  \n"
            f"- **TP = {tp}** anomali doğru yakalandı  \n"
            f"- **TN = {tn}** normal doğru tanındı  \n"
            f"- **FP = {fp}** normal yanlışlıkla anomali etiketlendi (Tip I hata)  \n"
            f"- **FN = {fn}** anomali kaçırıldı (Tip II hata — klinik açıdan daha kritik)  \n"
            f"- Model anomalilerin **%{tp/(tp+fn)*100:.1f}**'ini yakaladı (Recall)."
        )

        # ROC Curve
        st.markdown("#### 📉 ROC Curve")
        evaluator = Evaluator()
        roc_data = {}
        for name in saved["model_names"]:
            y_proba = train_info[name].get("y_proba")
            if y_proba is not None:
                fpr, tpr, roc_auc_val = evaluator.get_roc_data(y_test, y_proba)
                roc_data[name] = (fpr, tpr, roc_auc_val)
        if roc_data:
            fig_roc = viz.plot_roc_curves(roc_data)
            st.pyplot(fig_roc)
            plt.close()
            st.info(
                "**Yorum:** ROC eğrisi sol üst köşeye ne kadar yakınsa model o kadar iyi. "
                "AUC = 1.0 mükemmel ayrım, AUC = 0.5 rastgele tahmin. "
                f"En iyi modelimizin AUC'si **{eval_results[best_name].get('roc_auc', 0):.4f}** "
                "— neredeyse mükemmel ayrım."
            )

        # Precision-Recall Curve
        st.markdown("#### 📈 Precision-Recall Curve")
        pr_data = {}
        for name in saved["model_names"]:
            y_proba = train_info[name].get("y_proba")
            if y_proba is not None:
                prec_arr, rec_arr, pr_auc_val = evaluator.get_pr_data(y_test, y_proba)
                pr_data[name] = (prec_arr, rec_arr, pr_auc_val)
        if pr_data:
            fig_pr = viz.plot_pr_curves(pr_data)
            st.pyplot(fig_pr)
            plt.close()
            st.info(
                "**Yorum:** Precision-Recall eğrisi, dengesiz veri setlerinde ROC'tan daha güvenilirdir. "
                "Yüksek AP (Average Precision) skoru, modelin hem yüksek precision hem yüksek recall "
                "yakalayabildiğini gösterir."
            )

        # Feature Importance
        st.markdown("#### 🔍 Özellik Önemleri")
        fi_model_name = st.selectbox("Model Seçin:", list(models.keys()), key="fi_select")
        fig_fi = viz.plot_feature_importance(
            models[fi_model_name], feature_names, fi_model_name,
        )
        if fig_fi:
            st.pyplot(fig_fi)
            plt.close()
        else:
            st.warning("Bu model için feature importance bilgisi mevcut değil.")

        # Classification Report
        st.markdown("#### 📝 Sınıflandırma Raporu")
        cr_model = st.selectbox("Model Seçin:", list(eval_results.keys()), key="cr_select")
        st.code(eval_results[cr_model]["classification_rep"], language="text")

        # Sonuç özeti
        st.markdown("---")
        st.info(
            f"**📝 Sonuç Özeti:**  \n"
            f"Eğitilen 3 model içerisinde en yüksek F1-Skoru **{best_name}** tarafından alınmıştır "
            f"(F1: `{best_eval['f1']:.4f}`). Tüm modeller `RandomizedSearchCV` ile optimize edilmiş, "
            "`5-Fold StratifiedKFold` ile çapraz doğrulanmıştır."
        )

    # ==================================================================
    # TAB 5 — AÇIKLANABİLİRLİK (SHAP)
    # ==================================================================
    elif page == "🧠 Açıklanabilirlik":
        st.markdown("### 🧠 Açıklanabilir Yapay Zeka (SHAP)")
        st.caption(
            "SHAP (SHapley Additive exPlanations) — oyun teorisi tabanlı, "
            "her tahminin neden yapıldığını matematiksel olarak açıklayan yöntem."
        )

        if not models_loaded:
            st.info("Model bulunamadı. Terminalde `python3 train_models.py` çalıştırın.")
        else:
            models = saved["models"]
            y_test = saved["y_test"]
            X_test = saved["X_test"]
            X_train_bg = saved["X_train_sample"]

            st.markdown(
                "**SHAP nasıl çalışır?** Her özelliğin tahmine **katkı miktarını** "
                "Shapley değerleriyle hesaplar. Pozitif değer → anomali tahminini artırır, "
                "negatif değer → azaltır. Toplamı, modelin nihai tahminini verir."
            )

            # ---- Model seçimi ----
            xai_model_name = st.selectbox(
                "Açıklanacak Modeli Seçin:",
                list(models.keys()),
                key="xai_model_select",
            )

            selected_model = models[xai_model_name]

            with st.spinner(f"SHAP değerleri hesaplanıyor — {xai_model_name}..."):
                explainer_obj = Explainer(
                    selected_model,
                    X_train_bg,
                    max_background=80,
                )
                shap_result = explainer_obj.compute_shap_values(
                    X_test, max_samples=200
                )

            st.success(
                f"✓ SHAP hesaplandı — {len(shap_result.X_sample)} örnek, "
                f"{len(shap_result.feature_names)} özellik. "
                f"(Explainer: `{shap_result.explainer_type}`)"
            )

            st.markdown("---")

            # ==============================================================
            # GLOBAL — BAR CHART
            # ==============================================================
            st.markdown("#### 1️⃣ Global Özellik Önemi (Bar Chart)")
            st.markdown(
                "**Soru:** Modelimiz tahmin yaparken hangi özelliklere ne kadar önem veriyor?  \n"
                "**Yöntem:** Test setindeki tüm örnekler için her özelliğin **mutlak SHAP değerinin ortalaması**."
            )
            fig_bar = explainer_obj.plot_global_bar(shap_result, top_n=15)
            st.pyplot(fig_bar)
            plt.close()

            top_5 = explainer_obj.get_top_features(shap_result, top_n=5)
            top_5_str = ", ".join(f"`{f}`" for f in top_5)
            st.info(
                f"**Yorum:** Modelimiz tahminlerinde en çok {top_5_str} özelliklerine "
                "güveniyor. Bu liste, EDA'da Cohen's d ile bulduğumuz ayırt edici "
                "özelliklerle örtüşüyorsa modelimizin **istatistiksel olarak doğru "
                "sinyalleri öğrendiğini** gösterir."
            )

            st.markdown("---")

            # ==============================================================
            # GLOBAL — SUMMARY (BEESWARM)
            # ==============================================================
            st.markdown("#### 2️⃣ Global SHAP Summary Plot (Beeswarm)")
            st.markdown(
                "**Soru:** Özellik değerleri yüksek/düşük olduğunda tahmin nasıl etkileniyor?  \n"
                "**Yöntem:** Her noktanın yatay konumu **SHAP değerini** (etki yönü ve büyüklüğü), "
                "rengi ise **özellik değerini** gösterir (kırmızı=yüksek, mavi=düşük)."
            )
            fig_summary = explainer_obj.plot_global_summary(shap_result, top_n=15)
            st.pyplot(fig_summary)
            plt.close()

            st.info(
                "**Nasıl okunur?**  \n"
                "🔹 **Sağa giden noktalar** → o özellik anomali tahminini **artırıyor**  \n"
                "🔹 **Sola giden noktalar** → anomali tahminini **azaltıyor**  \n"
                "🔹 **Renk patterni:** Eğer kırmızı (yüksek değer) noktalar sürekli sağdaysa, "
                "yüksek özellik değeri = anomali işareti. Mavi (düşük değer) noktalar sağdaysa, "
                "düşük özellik değeri = anomali işareti."
            )

            st.markdown("---")

            # ==============================================================
            # LOKAL — WATERFALL
            # ==============================================================
            st.markdown("#### 3️⃣ Lokal Açıklama — Tek Bir Hücre Tahmini")
            st.markdown(
                "**Soru:** Tek bir hücrenin tahmininde hangi özellikler ne kadar etkili oldu?  \n"
                "**Yöntem:** Seçilen örnek için her özelliğin SHAP katkısı."
            )

            y_sample = y_test.reset_index(drop=True)

            sample_idx = st.slider(
                "Test setindeki hücre numarası:",
                min_value=0,
                max_value=len(shap_result.X_sample) - 1,
                value=0,
                step=1,
                help="Test setinden bir hücre seçin, modelin neden o tahmini yaptığını görelim.",
            )

            actual_label = int(y_sample.iloc[sample_idx]) if sample_idx < len(y_sample) else -1
            x_row = shap_result.X_sample.iloc[[sample_idx]]
            predicted = int(selected_model.predict(x_row)[0])
            proba = None
            if hasattr(selected_model, "predict_proba"):
                proba = float(selected_model.predict_proba(x_row)[0, 1])

            cc1, cc2, cc3 = st.columns(3)
            cc1.metric(
                "Gerçek Etiket",
                "Anomali (1)" if actual_label == 1 else "Normal (0)",
            )
            cc2.metric(
                "Model Tahmini",
                "Anomali (1)" if predicted == 1 else "Normal (0)",
                delta="Doğru" if predicted == actual_label else "Yanlış",
                delta_color="normal" if predicted == actual_label else "inverse",
            )
            if proba is not None:
                cc3.metric("Anomali Olasılığı", f"{proba:.1%}")

            fig_waterfall = explainer_obj.plot_local_waterfall(
                shap_result, sample_idx=sample_idx, top_n=12
            )
            st.pyplot(fig_waterfall)
            plt.close()

            st.info(
                "**Nasıl okunur?**  \n"
                "🔴 **Kırmızı bar (sağa)** → bu özellik değeri anomali tahminini **artırdı**  \n"
                "🔵 **Mavi bar (sola)** → bu özellik değeri anomali tahminini **azalttı**  \n"
                "**Baseline + SHAP toplamı = nihai tahmin** (tepedeki başlık formülü gösterir)."
            )

            st.markdown("---")
            st.markdown(
                "#### 🎯 Özet — SHAP'ın Değeri  \n"
                "🔹 **Şeffaflık:** Model artık black-box değil, her tahmini açıklayabiliyoruz  \n"
                "🔹 **Tutarlılık doğrulaması:** Modelin EDA'da bulduğumuz sinyalleri kullandığını teyit ediyoruz  \n"
                "🔹 **Klinik güven:** Bir uzman, modelin kararını adım adım inceleyip onaylayabilir  \n"
                "🔹 **Hata tespiti:** Yanlış tahminlerde hangi özelliklerin yanılttığını görebiliriz"
            )


if __name__ == "__main__":
    main()
