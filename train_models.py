"""
Offline Model Eğitim Scripti
============================
3 modeli (XGBoost, LightGBM, Random Forest) + multi-class hastalık modelini
eğitir, saved_models/ altına kaydeder.

Kullanım: python3 train_models.py
"""

import os
import joblib
import anomaly_detection as ad


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "archive")
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
RANDOM_STATE = 42

os.makedirs(SAVE_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("KAN HÜCRESİ ANOMALİ TESPİTİ — Model Eğitimi")
    print("=" * 60)

    # 1. Veri
    print("\n[1/4] Veri yükleniyor...")
    raw_df = ad.load_data(DATA_DIR)
    df_clean = ad.clean_data(raw_df)
    print(f"  {raw_df.shape[0]} satır, {raw_df.shape[1]} sütun")
    print(f"  Anomali oranı: {raw_df[ad.TARGET].mean():.2%}")

    # 2. Ön işleme
    print("\n[2/4] Ön işleme (one-hot + split + scale)...")
    pre = ad.preprocess(df_clean, random_state=RANDOM_STATE)
    print(f"  Train: {pre['X_train'].shape}, Test: {pre['X_test'].shape}")

    # 3. 3 modeli eğit
    print("\n[3/4] Model eğitimi (5-fold CV)...")
    train_results = {}
    eval_results = {}

    for name in ad.MODEL_NAMES:
        print(f"\n  → {name}")
        model = ad.build_model(name, random_state=RANDOM_STATE)
        tr = ad.train_with_cv(
            model, pre["X_train"], pre["y_train"], pre["X_test"],
            random_state=RANDOM_STATE,
        )
        ev = ad.evaluate(
            pre["y_test"], tr["y_pred"], tr["y_proba"],
            cv_mean=tr["cv_mean"], cv_std=tr["cv_std"],
        )
        train_results[name] = tr
        eval_results[name] = ev
        print(f"    F1: {ev['f1']:.4f}  ·  Acc: {ev['accuracy']:.4f}  "
              f"·  CV: {tr['cv_mean']:.4f} ± {tr['cv_std']:.4f}")

    comp_df = ad.comparison_table(eval_results)
    best_name = comp_df.index[0]
    print(f"\nEn iyi model: {best_name} (F1: {eval_results[best_name]['f1']:.4f})")

    # 4. Hastalık sınıflandırıcı
    print("\n[4/4] Hastalık sınıflandırıcı (multi-class)...")
    disease = ad.train_disease_classifier(raw_df, random_state=RANDOM_STATE)
    print(f"  Accuracy: {disease['accuracy']:.4f}")
    print(f"  Sınıflar: {disease['classes']}")

    # Kayıt
    print(f"\nKaydediliyor → {SAVE_DIR}/")
    for name, tr in train_results.items():
        safe = name.lower().replace(" ", "_")
        joblib.dump(tr["model"], os.path.join(SAVE_DIR, f"{safe}.joblib"))
        print(f"  ✓ {name}")

    joblib.dump({
        "scaler": pre["scaler"],
        "feature_names": pre["feature_names"],
        "numerical_cols": pre["numerical_cols"],
    }, os.path.join(SAVE_DIR, "preprocessor.joblib"))

    joblib.dump(disease, os.path.join(SAVE_DIR, "disease_classifier.joblib"))

    metadata = {
        "model_names": ad.MODEL_NAMES,
        "best_model": best_name,
        "X_test": pre["X_test"],
        "y_test": pre["y_test"],
        "X_train_sample": pre["X_train"].head(100),
        "feature_names": pre["feature_names"],
        "comparison_table": comp_df,
        "train_results": {n: {k: v for k, v in r.items() if k != "model"}
                          for n, r in train_results.items()},
        "eval_results": eval_results,
        "disease_classes": disease["classes"],
        "disease_accuracy": disease["accuracy"],
    }
    joblib.dump(metadata, os.path.join(SAVE_DIR, "metadata.joblib"))

    print(f"\n{'=' * 60}\nTAMAMLANDI · streamlit run app.py\n{'=' * 60}")


if __name__ == "__main__":
    main()
