"""
🩸 Kan Hücresi Anomali Tespiti — Streamlit Dashboard
====================================================
6 sekmeli arayüz: Veri Seti, EDA, Eğitim, Sonuçlar, Açıklanabilirlik,
Manuel Tahmin.

Tüm ML mantığı `anomaly_detection.py` içinde.
"""

import os
import warnings
import joblib
import matplotlib.pyplot as plt
import streamlit as st

import anomaly_detection as ad

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Kan Hücresi Anomali Tespiti",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(145deg, #0f172a 0%, #020617 100%); }
    .block-container { padding-top: 2rem !important; max-width: 1300px; }
    h1, h2, h3 { color: #f8fafc; }
    h2 { border-bottom: 2px solid rgba(56,189,248,0.2); padding-bottom: 8px; }
    div[data-testid="stMetric"] {
        background: rgba(30,41,59,0.7);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px; padding: 16px;
    }
    section[data-testid="stSidebar"] {
        background: rgba(15,23,42,0.95) !important;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "archive")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")


# ---------------------------------------------------------------------------
# Yükleme
# ---------------------------------------------------------------------------
@st.cache_resource
def load_saved():
    meta_path = os.path.join(MODELS_DIR, "metadata.joblib")
    if not os.path.exists(meta_path):
        return None
    meta = joblib.load(meta_path)
    meta["models"] = {}
    for name in meta["model_names"]:
        p = os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}.joblib")
        if os.path.exists(p):
            meta["models"][name] = joblib.load(p)
    pre_p = os.path.join(MODELS_DIR, "preprocessor.joblib")
    if os.path.exists(pre_p):
        meta["preprocessor"] = joblib.load(pre_p)
    dx_p = os.path.join(MODELS_DIR, "disease_classifier.joblib")
    if os.path.exists(dx_p):
        meta["disease"] = joblib.load(dx_p)
    return meta


@st.cache_data
def load_raw():
    return ad.load_data(DATA_DIR)


# ---------------------------------------------------------------------------
# Sayfalar
# ---------------------------------------------------------------------------
def page_dataset(raw_df):
    st.markdown("### 📋 Veri Seti Genel Bakış")
    s = ad.get_summary(raw_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Satır", f"{s['n_rows']:,}")
    c2.metric("Sütun", f"{s['n_cols']}")
    c3.metric("Anomali", f"{s['n_anomalies']:,}")
    c4.metric("Anomali Oranı", f"{s['anomaly_ratio']:.1%}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Eksik Değer", f"{s['missing_values']}")
    c6.metric("Duplikasyon", f"{s['duplicate_rows']}")
    c7.metric("Sayısal", f"{raw_df.select_dtypes(include='number').shape[1]}")
    c8.metric("Kategorik", f"{raw_df.select_dtypes(include='object').shape[1]}")

    st.markdown("#### 🔍 İlk 10 Satır")
    st.dataframe(raw_df.head(10), use_container_width=True, height=300)

    st.markdown("#### 🧹 Çıkarılan Sütunlar (Leakage + ID)")
    st.warning(
        "Eğitim öncesi şu sütunlar çıkarıldı:  \n"
        f"- **Leakage:** {', '.join(ad.LEAKAGE_COLS)}  \n"
        f"- **ID:** {', '.join(ad.ID_COLS)}"
    )

    st.markdown("#### 📈 Temizlenmiş Veri İstatistikleri")
    df_clean = ad.clean_data(raw_df)
    st.dataframe(df_clean.describe().T.style.format("{:.3f}"),
                 use_container_width=True, height=350)


def page_eda(raw_df):
    st.markdown("### 📊 Keşifsel Veri Analizi")
    st.markdown("Bu bölümde, kan hücresi veri setindeki anomali ve normal hücrelerin istatistiksel dağılımları ve değişkenler arası ilişkiler görselleştirilmiştir.")

    df_clean = ad.clean_data(raw_df)

    # 1. Sınıf Dağılımı
    st.markdown("---")
    st.markdown("#### 1️⃣ Sınıf Dağılımı")
    st.caption("Veri setindeki hücrelerin 'Normal' ve 'Anomali' olarak dağılımını gösterir. Verinin ne kadar dengesiz olduğunu anlamak için önemlidir.")
    fig1 = ad.plot_class_distribution_combined(raw_df)
    if fig1:
        st.pyplot(fig1); plt.close()
        st.info("Veri seti dengesiz bir dağılıma sahiptir. Anomali sınıfı veri setinin yaklaşık %32'sini oluştururken, Normal sınıf %68'lik bir çoğunluğa sahiptir. Bu dengesizlik, model eğitimi sırasında 'stratify' işlemi veya F1-Score gibi dengeli metriklerin kullanılmasını gerektirir.")

    # 2. Hastalık / Hücre Kategorisi Dağılımı
    st.markdown("---")
    st.markdown("#### 2️⃣ Hastalık / Hücre Kategorisi Dağılımı")
    st.caption("Hücrelerin teşhis konulmuş alt kategorilerini gösterir. 'Normal_RBC', 'Infection', 'Leukemia' gibi farklı kan ve anomali türlerinin veri setindeki yoğunluğunu ifade eder.")
    fig2 = ad.plot_disease_category_distribution(raw_df)
    if fig2:
        st.pyplot(fig2); plt.close()
        st.info("En yaygın normal hücre tipi 'Normal_RBC' iken, en sık görülen anomali kategorisi 'Infection' veya 'Anemia' olabilir. Bazı nadir anomalilerin öğrenilmesi, veri azlığı nedeniyle model için zorlayıcı olabilir.")

    # 3. Korelasyon Isı Haritası
    st.markdown("---")
    st.markdown("#### 3️⃣ Korelasyon Isı Haritası (Correlation Heatmap)")
    st.caption("Sayısal kan değerlerinin birbirleriyle olan doğrusal ilişkilerini (Pearson korelasyonu) ölçer.")
    fig3 = ad.plot_correlation_heatmap(raw_df)
    if fig3:
        st.pyplot(fig3); plt.close()
        st.info("Kırmızı alanlar güçlü pozitif, mavi alanlar güçlü negatif ilişkileri gösterir. Örneğin, hücre çapı ile hücre hacmi arasında genellikle yüksek korelasyon beklenir. Çok yüksek korelasyona sahip (multicollinearity) değişkenlerin varlığı, bazı modellerin yorumlanabilirliğini etkileyebilir.")

    # Top özellikleri hesapla (KDE ve Boxplot için)
    top_features = ad.get_top_features(df_clean, top_n=4)

    # 4. Normal vs Anomali - KDE
    st.markdown("---")
    st.markdown("#### 4️⃣ Normal vs Anomali — Özellik Dağılımları (KDE)")
    st.caption("Normal (mavi) ve Anomali (kırmızı) sınıfları arasındaki ayrımı en iyi yapan (Cohen's d skoru en yüksek olan) 4 özelliğin yoğunluk grafiği.")
    if top_features:
        fig4 = ad.plot_kde_grid(df_clean, top_features)
        if fig4:
            st.pyplot(fig4); plt.close()
            st.success(f"Yukarıdaki grafiklerde mavi ve kırmızı tepeciklerin (dağılımların) birbirinden ayrılması, bu özelliklerin anomalileri tespit etmede modele güçlü sinyaller verdiğini gösterir. Gösterilen özellikler: {', '.join(f'`{f}`' for f in top_features)}.")

    # 5. Boxplot
    st.markdown("---")
    st.markdown("#### 5️⃣ Kan Değerleri Kutu Grafiği (Boxplot)")
    st.caption("Aynı 4 önemli özelliğin istatistiksel özetini (medyan, çeyreklikler) ve aykırı değerleri (outliers) gösterir.")
    if top_features:
        fig5 = ad.plot_boxplot_grid(df_clean, top_features)
        if fig5:
            st.pyplot(fig5); plt.close()
            st.info("Kutu grafikleri, anomali sınıfında değerlerin nasıl daha geniş bir aralığa (varyansa) sahip olduğunu veya aykırı değerlerin daha sık görüldüğünü ortaya çıkarabilir. Modele standartlaştırma (StandardScaler) uygulamasının neden önemli olduğu buradan anlaşılabilir.")


def page_training(saved):
    st.markdown("### 🤖 Model Eğitimi")
    if not saved or not saved.get("models"):
        st.error("Model bulunamadı. `python3 train_models.py` çalıştırın.")
        return

    st.success(f"✅ {len(saved['models'])} model yüklendi · "
               "5-fold StratifiedKFold CV uygulandı.")

    st.markdown("#### Pipeline")
    st.markdown(
        "1. **Temizleme** — Leakage + ID sütunları çıkarıldı  \n"
        "2. **One-Hot Encoding** — Kategorikler sayısal  \n"
        "3. **Stratified Split** — %80 / %20  \n"
        "4. **StandardScaler** — sadece train'de fit  \n"
        "5. **5-Fold StratifiedKFold CV** — F1 skoru  \n"
        "6. **3 model:** XGBoost, LightGBM, Random Forest"
    )

    st.markdown("---")
    st.markdown("#### Modeller")
    for name in saved["model_names"]:
        ev = saved["eval_results"].get(name, {})
        tr = saved["train_results"].get(name, {})
        is_best = (name == saved.get("best_model"))
        title = f"🏆 {name}" if is_best else f"✅ {name}"
        with st.expander(title, expanded=is_best):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{ev.get('accuracy', 0):.4f}")
            m2.metric("Precision", f"{ev.get('precision', 0):.4f}")
            m3.metric("Recall", f"{ev.get('recall', 0):.4f}")
            m4.metric("F1-Score", f"{ev.get('f1', 0):.4f}")
            m5, m6, m7, m8 = st.columns(4)
            m5.metric("MCC", f"{ev.get('mcc', 0):.4f}")
            m6.metric("Kappa", f"{ev.get('kappa', 0):.4f}")
            if ev.get("roc_auc") is not None:
                m7.metric("ROC-AUC", f"{ev['roc_auc']:.4f}")
            if tr.get("cv_mean") is not None:
                m8.metric("CV Mean", f"{tr['cv_mean']:.4f} ± {tr['cv_std']:.4f}")


def page_results(saved):
    st.markdown("### 📈 Sonuçlar ve Karşılaştırma")
    if not saved or not saved.get("models"):
        st.info("Model bulunamadı."); return

    eval_results = saved["eval_results"]
    y_test = saved["y_test"]
    feature_names = saved["feature_names"]
    comp_df = saved["comparison_table"]
    models = saved["models"]
    train_info = saved["train_results"]
    best = saved["best_model"]
    best_ev = eval_results[best]

    st.success(f"🏆 **En İyi Model: {best}** · F1: `{best_ev['f1']:.4f}` · "
               f"Acc: `{best_ev['accuracy']:.4f}` · MCC: `{best_ev['mcc']:.4f}`")

    st.markdown("#### Karşılaştırma Tablosu")
    st.dataframe(comp_df.style.format("{:.4f}").highlight_max(axis=0, color="#1d4ed8"),
                 use_container_width=True)

    st.markdown("#### Performans Karşılaştırması")
    st.pyplot(ad.plot_metric_comparison(comp_df)); plt.close()

    st.markdown(f"#### Confusion Matrix — {best}")
    y_pred = train_info[best]["y_pred"]
    st.pyplot(ad.plot_confusion(y_test, y_pred, best)); plt.close()

    from sklearn.metrics import confusion_matrix as _cm
    tn, fp, fn, tp = _cm(y_test, y_pred).ravel()
    st.info(
        f"**TP={tp}** · **TN={tn}** · **FP={fp}** · **FN={fn}**  \n"
        f"Recall (anomali yakalama): **%{tp/(tp+fn)*100:.1f}**"
    )

    st.markdown("#### ROC Eğrileri")
    roc_data = {}
    for name in saved["model_names"]:
        proba = train_info[name].get("y_proba")
        if proba is not None:
            from sklearn.metrics import roc_curve as _rc, auc as _auc
            fpr, tpr, _ = _rc(y_test, proba)
            roc_data[name] = (fpr, tpr, _auc(fpr, tpr))
    if roc_data:
        st.pyplot(ad.plot_roc(roc_data)); plt.close()

    st.markdown("#### Precision-Recall Eğrileri")
    pr_data = {}
    for name in saved["model_names"]:
        proba = train_info[name].get("y_proba")
        if proba is not None:
            from sklearn.metrics import precision_recall_curve as _prc, average_precision_score as _ap
            pr_arr, rc_arr, _ = _prc(y_test, proba)
            pr_data[name] = (pr_arr, rc_arr, _ap(y_test, proba))
    if pr_data:
        st.pyplot(ad.plot_pr(pr_data)); plt.close()

    st.markdown("#### Özellik Önemleri")
    fi_name = st.selectbox("Model:", list(models.keys()), key="fi")
    fig = ad.plot_feature_importance(models[fi_name], feature_names)
    if fig:
        st.pyplot(fig); plt.close()



def page_xai(saved):
    st.markdown("### 🧠 Açıklanabilirlik (SHAP)")
    if not saved or not saved.get("models"):
        st.info("Model bulunamadı."); return

    models = saved["models"]
    X_test = saved["X_test"]
    y_test = saved["y_test"]

    name = st.selectbox("Model seçin:", list(models.keys()))
    model = models[name]

    with st.spinner(f"SHAP hesaplanıyor — {name}..."):
        # Hız için ilk 200 örnek
        X_sample = X_test.head(200) if len(X_test) > 200 else X_test
        result = ad.compute_shap(model, X_sample, saved["X_train_sample"])

    st.success(f"✓ {len(X_sample)} örnek üzerinde SHAP hesaplandı")

    st.markdown("#### 1️⃣ Global Özellik Önemi")
    st.pyplot(ad.plot_shap_bar(result, top_n=15)); plt.close()

    st.markdown("#### 2️⃣ Beeswarm Summary")
    st.caption("Sağ: anomaliyi artırır · Sol: azaltır · Renk: özellik değeri")
    st.pyplot(ad.plot_shap_summary(result, top_n=15)); plt.close()

    st.markdown("#### 3️⃣ Örnek Vaka Analizleri (Lokal Açıklama)")
    
    y_arr = y_test.reset_index(drop=True)
    preds = model.predict(X_sample)
    
    tp_idx, tn_idx, err_idx = None, None, None
    for i in range(len(X_sample)):
        actual = int(y_arr.iloc[i]) if i < len(y_arr) else -1
        pred = int(preds[i])
        
        if actual == 1 and pred == 1 and tp_idx is None:
            tp_idx = i
        elif actual == 0 and pred == 0 and tn_idx is None:
            tn_idx = i
        elif actual != pred and err_idx is None:
            err_idx = i
            
        if tp_idx is not None and tn_idx is not None and err_idx is not None:
            break
            
    cases = [
        (tp_idx, "### Vaka 1: Başarılı Anomali Tespiti (True Positive)"),
        (tn_idx, "### Vaka 2: Başarılı Normal Hücre Tespiti (True Negative)"),
        (err_idx, "### Vaka 3: Model Yanılgısı (Hata Analizi)")
    ]
    
    for idx, title in cases:
        if idx is None:
            continue
            
        st.markdown(title)
        actual = int(y_arr.iloc[idx]) if idx < len(y_arr) else -1
        pred = int(preds[idx])
        proba = float(model.predict_proba(X_sample.iloc[[idx]])[0, 1])
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Gerçek", "Anomali" if actual == 1 else "Normal")
        c2.metric("Tahmin", "Anomali" if pred == 1 else "Normal",
                  delta="Doğru" if pred == actual else "Yanlış",
                  delta_color="normal" if pred == actual else "inverse")
        c3.metric("Anomali Olasılığı", f"{proba:.1%}")
        st.pyplot(ad.plot_shap_waterfall(result, idx, top_n=12)); plt.close()
        st.markdown("---")


def page_predict(raw_df, saved):
    st.markdown("### 🔬 Manuel Tahmin")
    st.caption("Hücre özelliklerini gir → anomali ve hastalık tahmini al.")

    if not saved or not saved.get("models") or "preprocessor" not in saved:
        st.error("Model bulunamadı. `python3 train_models.py` çalıştırın."); return

    df_clean = ad.clean_data(raw_df)
    df_features = df_clean.drop(columns=[ad.TARGET], errors="ignore")
    num_cols = df_features.select_dtypes(include="number").columns.tolist()
    cat_cols = df_features.select_dtypes(include="object").columns.tolist()

    model_names = list(saved["models"].keys())
    default = saved.get("best_model", model_names[0])
    chosen = st.selectbox("Tahmin için model:", model_names,
                          index=model_names.index(default))
    model = saved["models"][chosen]

    st.markdown("---")
    st.markdown("#### 📝 Girdi (medyanlar ön doldurulmuş)")

    with st.form("predict_form"):
        input_dict = {}
        st.markdown("**Sayısal Özellikler**")
        cols = st.columns(3)
        for i, c in enumerate(num_cols):
            s = df_features[c].dropna()
            v_min, v_max = float(s.min()), float(s.max())
            v_med = float(s.median())
            rng = max(v_max - v_min, 1e-6)
            step = float(rng / 100) if rng < 50 else 1.0
            with cols[i % 3]:
                input_dict[c] = st.number_input(
                    c, min_value=float(v_min - rng), max_value=float(v_max + rng),
                    value=v_med, step=step,
                    format="%.4f" if rng < 10 else "%.2f",
                )
        if cat_cols:
            st.markdown("**Kategorik Özellikler**")
            ccols = st.columns(min(3, len(cat_cols)))
            for i, c in enumerate(cat_cols):
                opts = sorted(df_features[c].dropna().unique().tolist())
                with ccols[i % len(ccols)]:
                    input_dict[c] = st.selectbox(c, opts)

        submitted = st.form_submit_button("🔍 Tahmin Et", use_container_width=True)

    if not submitted:
        return

    # Anomali modeli tahmini (ham)
    try:
        raw_pred, raw_proba = ad.predict_single(input_dict, model, saved["preprocessor"])
    except Exception as e:
        st.error(f"Tahmin hatası: {e}"); return

    # Hastalık sınıflandırıcı tahmini
    disease_label = None
    disease_top3 = None
    is_normal_category = False
    normal_total_proba = 0.0
    if "disease" in saved:
        try:
            disease_label, disease_top3, normal_total_proba = ad.predict_disease(
                input_dict, saved["disease"]
            )
            is_normal_category = disease_label.lower().startswith("normal")
        except Exception as e:
            st.warning(f"Hastalık tahmini yapılamadı: {e}")

    # Nihai anomali kararı: hastalık kategorisine göre belirle
    # Normal_WBC, Normal_RBC, Normal_Platelet → normal hücre
    # Anemia, Leukemia, Infection, Sickle_Cell_Anemia, Artefact → anomali
    if disease_label is not None:
        final_pred = 0 if is_normal_category else 1
        anomaly_proba = 1.0 - normal_total_proba
    else:
        final_pred = raw_pred
        anomaly_proba = raw_proba

    st.markdown("---")
    st.markdown("#### 🎯 Anomali Tahmini")
    c1, c2, c3 = st.columns(3)
    if final_pred == 1:
        c1.error("**ANOMALİ TESPİT EDİLDİ**")
    else:
        c1.success("**NORMAL HÜCRE**")
    c2.metric("Tahmin (0/1)", final_pred)
    if anomaly_proba is not None:
        c3.metric("Anomali Olasılığı", f"{anomaly_proba:.1%}")
        st.progress(min(max(anomaly_proba, 0.0), 1.0))

    # Hastalık kategorisi gösterimi
    if disease_label is not None:
        st.markdown("---")
        st.markdown("#### 🧬 Hastalık / Hücre Kategorisi")

        # Kategori etiketlerinin Türkçe açıklamaları
        CATEGORY_DESCRIPTIONS = {
            "Normal_WBC": "Normal beyaz kan hücresi (lökosit). Sağlıklı bağışıklık sistemi hücresi.",
            "Normal_RBC": "Normal kırmızı kan hücresi (eritrosit). Sağlıklı oksijen taşıyıcı hücre.",
            "Normal_Platelet": "Normal trombosit. Sağlıklı pıhtılaşma hücresi.",
            "Anemia": "Anemi — kırmızı kan hücrelerinin sayı veya fonksiyon bozukluğu. Oksijen taşıma kapasitesi düşük.",
            "Leukemia": "Lösemi — beyaz kan hücrelerinin kontrolsüz çoğalması. Kan kanseri türü.",
            "Infection": "Enfeksiyon belirtisi — bağışıklık sistemi aktif yanıt veriyor. Hücre morfolojisinde değişiklik.",
            "Sickle_Cell_Anemia": "Orak hücreli anemi — genetik bir hastalık. Kırmızı kan hücreleri orak şeklinde deforme.",
            "Artefact": "Artefakt — laboratuvar sürecinde oluşan yapay bozulma. Gerçek hücre anomalisi değil.",
        }

        d1, d2 = st.columns([1, 2])
        if is_normal_category:
            d1.success(f"**{disease_label}**")
        else:
            d1.error(f"**{disease_label}**")

        # Kategori açıklaması
        desc = CATEGORY_DESCRIPTIONS.get(disease_label, "")
        if desc:
            d2.info(f"💬 {desc}")

        # Sonuç özeti
        n_classes = len(saved['disease']['classes'])
        acc = saved['disease']['accuracy']
        if is_normal_category:
            st.success(
                f"✅ Bu hücre **{disease_label}** olarak sınıflandırıldı ve **normal** bir hücredir. "
                f"Anomali tespit edilmedi."
            )
        else:
            st.error(
                f"⚠️ Bu hücre **{disease_label}** olarak sınıflandırıldı ve bu bir **anomali** durumudur. "
                f"Detaylı tıbbi değerlendirme önerilir."
            )

        st.markdown("**Top 3 olası kategori:**")
        for lbl, p in disease_top3:
            lbl_desc = CATEGORY_DESCRIPTIONS.get(lbl, "")
            short = lbl_desc.split(".")[0] + "." if lbl_desc else ""
            st.write(f"- `{lbl}` — **{p:.1%}** {('· _' + short + '_') if short else ''}")
            st.progress(min(max(p, 0.0), 1.0))

        with st.expander("ℹ️ Model bilgisi"):
            st.markdown(
                f"Bu tahmin, **{n_classes} farklı hücre/hastalık kategorisi** üzerinde "
                f"eğitilmiş bir XGBoost sınıflandırıcı tarafından yapılmıştır.  \n"
                f"Modelin test verisi üzerindeki doğruluk oranı **%{acc*100:.1f}**'dır "
                f"(yani test örneklerinin %{acc*100:.1f}'unu doğru sınıflandırmıştır).  \n\n"
                f"**Kategoriler:** {', '.join(f'`{c}`' for c in saved['disease']['classes'])}"
            )

    with st.expander("Modele giden ham girdi"):
        st.json(input_dict)


# ---------------------------------------------------------------------------
# Ana
# ---------------------------------------------------------------------------
def main():
    saved = load_saved()
    raw_df = load_raw()

    with st.sidebar:
        st.markdown("## 🩸 Kan Hücresi Anomali Tespiti")
        st.markdown("---")
        page = st.radio(
            "Sayfa",
            ["📋 Veri Seti", "📊 Keşifsel Analiz",
             "📈 Eğitim Sonuçları", "🧠 Açıklanabilirlik", "🔬 Manuel Tahmin"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        if not (saved and saved.get("models")):
            st.error("Model yok. `python3 train_models.py`")

    st.markdown("# 🩸 Kan Hücresi Anomali Tespiti")

    if page == "📋 Veri Seti":
        page_dataset(raw_df)
    elif page == "📊 Keşifsel Analiz":
        page_eda(raw_df)
    elif page == "📈 Eğitim Sonuçları":
        page_results(saved)
    elif page == "🧠 Açıklanabilirlik":
        page_xai(saved)
    elif page == "🔬 Manuel Tahmin":
        page_predict(raw_df, saved)


if __name__ == "__main__":
    main()
