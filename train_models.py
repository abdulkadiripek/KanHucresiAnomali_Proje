"""
=============================================================================
Offline Model Eğitim Scripti
=============================================================================
Bu script bir kez çalıştırılır, 3 modeli eğitir (XGBoost, LightGBM, RF)
ve tüm sonuçları saved_models/ altına kaydeder.

Streamlit uygulaması bu kayıtlı modelleri yükleyerek çalışır.

Kullanım:
    python3 train_models.py
=============================================================================
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from core import DataLoader, Preprocessor, ModelFactory, Evaluator

# ============================================================================
# KONFIGÜRASYON
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "archive")
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_NAMES = ["XGBoost", "LightGBM", "Random Forest"]
TEST_SIZE = 0.20
N_CV_FOLDS = 5
RANDOM_STATE = 42


def main():
    print("=" * 70)
    print("OFFLINE MODEL EĞİTİM — Kan Hücresi Anomali Tespiti")
    print("=" * 70)

    # ================================================================
    # 1. Veri Yükleme ve Temizleme
    # ================================================================
    print("\n[1/5] Veri yükleniyor...")
    loader = DataLoader(data_dir=DATA_DIR)
    raw_df = loader.load()
    df_clean = loader.clean(raw_df)
    summary = loader.get_summary(raw_df)
    print(f"  Veri seti: {summary['n_rows']} satır, {summary['n_cols']} sütun")
    print(f"  Anomali oranı: {summary['anomaly_ratio']:.2%}")

    # ================================================================
    # 2. Preprocessing
    # ================================================================
    print("\n[2/5] Veri ön işleme...")
    preprocessor = Preprocessor(
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        apply_smote=False,
    )
    split = preprocessor.process(df_clean)
    print(f"  Train: {split.X_train.shape[0]} satır")
    print(f"  Test:  {split.X_test.shape[0]} satır")
    print(f"  Özellik: {split.X_train.shape[1]}")

    # ================================================================
    # 3. Model Eğitimi (CV + Tuning)
    # ================================================================
    print("\n[3/5] Model eğitimi başlıyor...")
    factory = ModelFactory(random_state=RANDOM_STATE, n_cv_folds=N_CV_FOLDS)
    evaluator = Evaluator()

    train_results = {}

    for name in MODEL_NAMES:
        print(f"\n  >>> {name}")

        # Hyperparameter tuning
        print(f"      RandomizedSearchCV (20 iterasyon)...")
        try:
            result = factory.tune_hyperparameters(
                name,
                split.X_train, split.y_train,
                split.X_test,
                n_iter=20,
                scoring="f1",
            )
            display_name = result.model_name  # "XGBoost (Tuned)"
            print(f"      En iyi parametreler: {result.best_params}")
        except (ValueError, KeyError):
            # Tune desteklemiyorsa default ile eğit
            print(f"      Tune desteklenmiyor, default parametrelerle eğitiliyor...")
            model = factory.create_model(name)
            result = factory.train(
                name, model,
                split.X_train, split.y_train,
                split.X_test, split.y_test,
                run_cv=True,
            )
            display_name = name

        # Evaluate
        eval_result = evaluator.evaluate(
            model_name=display_name,
            y_true=split.y_test,
            y_pred=result.y_pred,
            y_proba=result.y_proba,
            cv_mean=result.cv_mean,
            cv_std=result.cv_std,
        )

        train_results[display_name] = result
        print(f"      F1-Score: {eval_result.f1:.4f}")
        print(f"      Accuracy: {eval_result.accuracy:.4f}")
        if result.cv_mean is not None:
            std_str = f" ± {result.cv_std:.4f}" if result.cv_std is not None else ""
            print(f"      CV Mean:  {result.cv_mean:.4f}{std_str}")

    # ================================================================
    # 4. En İyi Model
    # ================================================================
    print(f"\n{'=' * 70}")
    best_name = evaluator.get_best_model("F1-Score")
    best_eval = evaluator.results[best_name]
    print(f"En iyi model: {best_name} (F1: {best_eval.f1:.4f})")

    # Karşılaştırma tablosu
    comp_df = evaluator.get_comparison_table()
    print(f"\n{comp_df.to_string(float_format='{:.4f}'.format)}")

    # ================================================================
    # 5. Kaydet
    # ================================================================
    print(f"\n[5/5] Modeller kaydediliyor → {SAVE_DIR}/")

    # Her model ayrı dosya
    for name, result in train_results.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_path = os.path.join(SAVE_DIR, f"{safe_name}.joblib")
        joblib.dump(result.model, model_path)
        print(f"  ✓ {name} → {model_path}")

    # Preprocessor (scaler + feature names)
    preprocessor_path = os.path.join(SAVE_DIR, "preprocessor.joblib")
    joblib.dump({
        "scaler": split.scaler,
        "feature_names": split.feature_names,
        "numerical_cols": split.numerical_cols,
        "categorical_cols": split.categorical_cols,
    }, preprocessor_path)
    print(f"  ✓ Preprocessor → {preprocessor_path}")

    # Metadata (eval sonuçları, test verileri, tahminler)
    metadata = {
        "model_names": list(train_results.keys()),
        "best_model": best_name,
        "X_test": split.X_test,
        "y_test": split.y_test,
        "X_train_sample": split.X_train.head(100),  # SHAP background için
        "feature_names": split.feature_names,
        "comparison_table": comp_df,
        "train_results": {},
        "eval_results": {},
    }

    for name, result in train_results.items():
        metadata["train_results"][name] = {
            "y_pred": result.y_pred,
            "y_proba": result.y_proba,
            "cv_scores": result.cv_scores,
            "cv_mean": result.cv_mean,
            "cv_std": result.cv_std,
            "best_params": result.best_params,
        }

    for name, eval_res in evaluator.results.items():
        metadata["eval_results"][name] = {
            "accuracy": eval_res.accuracy,
            "balanced_accuracy": eval_res.balanced_accuracy,
            "precision": eval_res.precision,
            "recall": eval_res.recall,
            "f1": eval_res.f1,
            "mcc": eval_res.mcc,
            "kappa": eval_res.kappa,
            "log_loss_val": eval_res.log_loss_val,
            "roc_auc": eval_res.roc_auc,
            "pr_auc": eval_res.pr_auc,
            "confusion_mat": eval_res.confusion_mat,
            "classification_rep": eval_res.classification_rep,
            "cv_mean": eval_res.cv_mean,
            "cv_std": eval_res.cv_std,
        }

    metadata_path = os.path.join(SAVE_DIR, "metadata.joblib")
    joblib.dump(metadata, metadata_path)
    print(f"  ✓ Metadata → {metadata_path}")

    # ================================================================
    # SONUÇ
    # ================================================================
    print(f"\n{'=' * 70}")
    print("TAMAMLANDI")
    print(f"{'=' * 70}")
    print(f"  {len(MODEL_NAMES)} model eğitildi ve kaydedildi.")
    print(f"  En iyi: {best_name} (F1: {best_eval.f1:.4f})")
    print(f"  Kayıt dizini: {SAVE_DIR}/")
    print(f"\n  Streamlit'i başlatmak için:")
    print(f"    streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
