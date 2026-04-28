"""
=============================================================================
Blood Cell Anomaly Detection — Klasik Makine Öğrenmesi ile Anomali Tespiti
=============================================================================
Bu script, kan hücresi veri setindeki anomalileri tespit etmek için
XGBoost, Random Forest ve LightGBM algoritmalarını kullanır.

NOT: Derin öğrenme (Neural Network, PyTorch, TensorFlow vb.) KULLANILMAMISTIR.
     Sadece klasik ML algoritmaları kullanılmıştır.
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Gereksiz uyarıları bastır
warnings.filterwarnings("ignore")

# Matplotlib Türkçe karakter desteği ve genel stil ayarları
plt.rcParams["figure.dpi"] = 150
sns.set_theme(style="whitegrid", font_scale=1.1)

# ============================================================================
# 1. VERİ YÜKLEME VE TEMİZLEME
# ============================================================================
print("=" * 70)
print("ADIM 1: Veri Yükleme ve Temizleme")
print("=" * 70)

# CSV dosyasını oku
DATA_PATH = os.path.join(os.path.dirname(__file__), "archive", "blood_cell_anomaly_detection.csv")
df = pd.read_csv(DATA_PATH)
print(f"Orijinal veri seti boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")

# Hedef değişken bilgisi
print(f"\nHedef değişken (anomaly_label) dağılımı:")
print(df["anomaly_label"].value_counts().to_string())
print(f"Anomali oranı: {df['anomaly_label'].mean():.2%}")

# --- cell_id sütununu kaldır (modellemeye katkısı yok) ---
df.drop(columns=["cell_id"], inplace=True)
print("\n✓ 'cell_id' sütunu kaldırıldı.")

# --- Data Leakage'a neden olacak sütunları kaldır ---
leakage_columns = [
    "disease_category",
    "cell_type",
    "cytodiffusion_anomaly_score",
    "cytodiffusion_classification_confidence",
    "labeller_confidence_score",
]
df.drop(columns=leakage_columns, inplace=True)
print(f"✓ Data Leakage sütunları kaldırıldı: {leakage_columns}")
print(f"Temizlenmiş veri seti boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")

# ============================================================================
# 2. VERİ ÖN İŞLEME (PREPROCESSING)
# ============================================================================
print("\n" + "=" * 70)
print("ADIM 2: Veri Ön İşleme (Preprocessing)")
print("=" * 70)

# Hedef (y) ve özellik matrisini (X) ayır
y = df["anomaly_label"]
X = df.drop(columns=["anomaly_label"])

# --- Kategorik ve sayısal sütunları tespit et ---
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()

print(f"\nKategorik sütunlar ({len(categorical_cols)}): {categorical_cols}")
print(f"Sayısal sütunlar   ({len(numerical_cols)}): {numerical_cols}")

# --- One-Hot Encoding (kategorik sütunlar) ---
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
print(f"\n✓ One-Hot Encoding uygulandı.")
print(f"  Yeni özellik sayısı: {X.shape[1]}")

# --- Train / Test Split (%80 / %20, stratified) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n✓ Veri bölündü: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

# --- StandardScaler (sadece sayısal sütunlara uygula) ---
# Not: Scaler sadece train üzerinde fit edilir, test'e ayrıca transform uygulanır.
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
print("✓ StandardScaler ile sayısal özellikler ölçeklendi (fit: train, transform: test).")

# ============================================================================
# 3. MODEL EĞİTİMİ (3 Klasik ML Algoritması)
# ============================================================================
print("\n" + "=" * 70)
print("ADIM 3: Model Eğitimi")
print("=" * 70)

# Modelleri tanımla
models = {
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        max_depth=-1,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    ),
}

# Her modeli eğit ve sonuçları sakla
results = {}
predictions = {}

for name, model in models.items():
    print(f"\n>>> {name} eğitiliyor...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
    }
    predictions[name] = y_pred
    print(f"    ✓ {name} eğitimi tamamlandı.")

# ============================================================================
# 4. DEĞERLENDİRME (EVALUATION)
# ============================================================================
print("\n" + "=" * 70)
print("ADIM 4: Değerlendirme (Evaluation)")
print("=" * 70)

# --- Her model için metrikleri ekrana yazdır ---
for name, metrics in results.items():
    print(f"\n{'─' * 50}")
    print(f"  Model: {name}")
    print(f"{'─' * 50}")
    print(f"  Accuracy  : {metrics['Accuracy']:.4f}")
    print(f"  Precision : {metrics['Precision']:.4f}")
    print(f"  Recall    : {metrics['Recall']:.4f}")
    print(f"  F1-Score  : {metrics['F1-Score']:.4f}")

# --- Karşılaştırma tablosu ---
print(f"\n{'=' * 70}")
print("MODEL KARŞILAŞTIRMA TABLOSU")
print(f"{'=' * 70}")

comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.sort_values("F1-Score", ascending=False)
print(comparison_df.to_string(float_format="{:.4f}".format))

# --- En iyi modeli belirle (F1-Score'a göre) ---
best_model_name = comparison_df.index[0]
best_f1 = comparison_df.loc[best_model_name, "F1-Score"]
print(f"\n🏆 En iyi model: {best_model_name} (F1-Score: {best_f1:.4f})")

# --- En iyi modelin Confusion Matrix'ini çizdir ---
cm = confusion_matrix(y_test, predictions[best_model_name])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal (0)", "Anomaly (1)"],
    yticklabels=["Normal (0)", "Anomaly (1)"],
    linewidths=1,
    linecolor="white",
    annot_kws={"size": 16, "weight": "bold"},
    ax=ax,
)
ax.set_xlabel("Tahmin Edilen (Predicted)", fontsize=13)
ax.set_ylabel("Gerçek (Actual)", fontsize=13)
ax.set_title(f"Confusion Matrix — {best_model_name}\n(F1-Score: {best_f1:.4f})", fontsize=14, weight="bold")
plt.tight_layout()

# Confusion Matrix'i kaydet
output_path = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")
fig.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"\n✓ Confusion Matrix kaydedildi: {output_path}")
plt.show()

# --- Özet ---
print(f"\n{'=' * 70}")
print("SONUÇ ÖZETİ")
print(f"{'=' * 70}")
print(f"Toplam {len(models)} klasik ML modeli eğitildi ve test seti üzerinde değerlendirildi.")
print(f"En yüksek F1-Score'a sahip model: {best_model_name} ({best_f1:.4f})")
print(f"Tüm modeller sadece hücrenin fiziksel/kimyasal özellikleri üzerinden")
print(f"anomali tespiti yapmıştır. Data Leakage önlenmiştir.")
print(f"Hiçbir derin öğrenme yöntemi kullanılmamıştır.")
print("=" * 70)
