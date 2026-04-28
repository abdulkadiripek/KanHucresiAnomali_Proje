"""
=============================================================================
Blood Cell Anomaly Detection — CLI Script (Modüler)
=============================================================================
Kan hücresi veri setindeki anomalileri tespit etmek için klasik ML
algoritmalarını kullanan bağımsız CLI scripti.

Derin öğrenme KULLANILMAMISTIR. Sadece klasik ML algoritmaları.

Kullanım:
    python anomaly_detection.py

Modüler core/ paketi kullanır:
    - DataLoader  : Veri yükleme ve temizleme
    - Preprocessor: Encoding, scaling, split, SMOTE
    - ModelFactory: Model oluşturma ve eğitim
    - Evaluator   : Metrik hesaplama ve raporlama
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Core modüller
from core import DataLoader, Preprocessor, ModelFactory, Evaluator

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
plt.rcParams["figure.dpi"] = 150
sns.set_theme(style="whitegrid", font_scale=1.1)


# ============================================================================
# KONFIGÜRASYON
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "archive")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    """Ana çalıştırma fonksiyonu."""

    # ================================================================
    # ADIM 1: Veri Yükleme ve Temizleme
    # ================================================================
    print("=" * 70)
    print("ADIM 1: Veri Yükleme ve Temizleme")
    print("=" * 70)

    loader = DataLoader(data_dir=DATA_DIR)
    raw_df = loader.load()
    summary = loader.get_summary(raw_df)

    print(f"Orijinal veri seti: {summary['n_rows']} satır, {summary['n_cols']} sütun")
    print(f"Anomali oranı: {summary['anomaly_ratio']:.2%}")
    print(f"Eksik değer: {summary['missing_values']}")

    df_clean = loader.clean(raw_df)
    print(f"\n✓ Temizlenmiş veri seti: {df_clean.shape[0]} satır, {df_clean.shape[1]} sütun")

    # ================================================================
    # ADIM 2: Veri Ön İşleme
    # ================================================================
    print(f"\n{'=' * 70}")
    print("ADIM 2: Veri Ön İşleme (Preprocessing)")
    print("=" * 70)

    preprocessor = Preprocessor(test_size=0.20, apply_smote=False)
    split = preprocessor.process(df_clean)

    print(f"Kategorik sütunlar: {split.categorical_cols}")
    print(f"Sayısal sütunlar: {split.numerical_cols}")
    print(f"\n✓ Train: {split.X_train.shape[0]} satır, Test: {split.X_test.shape[0]} satır")
    print(f"✓ Özellik sayısı: {split.X_train.shape[1]}")

    # ================================================================
    # ADIM 3: Model Eğitimi (Cross-Validation ile)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("ADIM 3: Model Eğitimi (5-Fold Cross-Validation)")
    print("=" * 70)

    factory = ModelFactory(n_cv_folds=5)
    evaluator = Evaluator()

    model_names = ["XGBoost", "Random Forest", "LightGBM"]
    train_results = {}

    for name in model_names:
        print(f"\n>>> {name} eğitiliyor...")
        model = factory.create_model(name)
        result = factory.train(
            name, model,
            split.X_train, split.y_train,
            split.X_test, split.y_test,
            run_cv=True,
        )
        train_results[name] = result

        evaluator.evaluate(
            model_name=name,
            y_true=split.y_test,
            y_pred=result.y_pred,
            y_proba=result.y_proba,
            cv_mean=result.cv_mean,
            cv_std=result.cv_std,
        )
        print(f"    ✓ {name} eğitimi tamamlandı.")
        if result.cv_mean is not None:
            print(f"    CV F1-Score: {result.cv_mean:.4f} ± {result.cv_std:.4f}")

    # ================================================================
    # ADIM 4: Değerlendirme
    # ================================================================
    print(f"\n{'=' * 70}")
    print("ADIM 4: Değerlendirme")
    print("=" * 70)

    for name, eval_res in evaluator.results.items():
        print(f"\n{'─' * 55}")
        print(f"  Model: {name}")
        print(f"{'─' * 55}")
        print(f"  Accuracy        : {eval_res.accuracy:.4f}")
        print(f"  Balanced Acc.   : {eval_res.balanced_accuracy:.4f}")
        print(f"  Precision       : {eval_res.precision:.4f}")
        print(f"  Recall          : {eval_res.recall:.4f}")
        print(f"  F1-Score        : {eval_res.f1:.4f}")
        print(f"  MCC             : {eval_res.mcc:.4f}")
        print(f"  Cohen's Kappa   : {eval_res.kappa:.4f}")
        if eval_res.roc_auc is not None:
            print(f"  ROC-AUC         : {eval_res.roc_auc:.4f}")
        if eval_res.pr_auc is not None:
            print(f"  PR-AUC          : {eval_res.pr_auc:.4f}")

    # Karşılaştırma tablosu
    print(f"\n{'=' * 70}")
    print("MODEL KARŞILAŞTIRMA TABLOSU")
    print("=" * 70)

    comp_df = evaluator.get_comparison_table()
    print(comp_df.to_string(float_format="{:.4f}".format))

    # En iyi model
    best_name = evaluator.get_best_model("F1-Score")
    best_eval = evaluator.results[best_name]
    print(f"\n🏆 En iyi model: {best_name} (F1-Score: {best_eval.f1:.4f})")

    # Classification report
    print(f"\n{'=' * 70}")
    print(f"SINIFLANDIRMA RAPORU — {best_name}")
    print("=" * 70)
    print(best_eval.classification_rep)

    # ================================================================
    # ADIM 5: Confusion Matrix Kaydet
    # ================================================================
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(split.y_test, train_results[best_name].y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal (0)", "Anomaly (1)"],
        yticklabels=["Normal (0)", "Anomaly (1)"],
        linewidths=1, linecolor="white",
        annot_kws={"size": 16, "weight": "bold"},
        ax=ax,
    )
    ax.set_xlabel("Tahmin Edilen (Predicted)", fontsize=13)
    ax.set_ylabel("Gerçek (Actual)", fontsize=13)
    ax.set_title(
        f"Confusion Matrix — {best_name}\n(F1-Score: {best_eval.f1:.4f})",
        fontsize=14, weight="bold",
    )
    plt.tight_layout()

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Confusion Matrix kaydedildi: {cm_path}")

    # ================================================================
    # Model kaydet
    # ================================================================
    model_path = os.path.join(OUTPUT_DIR, f"{best_name.lower().replace(' ', '_')}_best.joblib")
    ModelFactory.save_model(train_results[best_name].model, model_path)
    print(f"✓ En iyi model kaydedildi: {model_path}")

    # ================================================================
    # SONUÇ
    # ================================================================
    print(f"\n{'=' * 70}")
    print("SONUÇ ÖZETİ")
    print("=" * 70)
    print(f"Toplam {len(model_names)} klasik ML modeli eğitildi.")
    print(f"5-Fold StratifiedKFold Cross-Validation uygulandı.")
    print(f"En yüksek F1-Score: {best_name} ({best_eval.f1:.4f})")
    print(f"Hiçbir derin öğrenme yöntemi kullanılmamıştır.")
    print("=" * 70)


if __name__ == "__main__":
    main()
