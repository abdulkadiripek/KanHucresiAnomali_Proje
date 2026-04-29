import os
import joblib
import warnings
import pandas as pd
import numpy as np
from core import DataLoader

warnings.filterwarnings("ignore")

def predict_anomaly(input_dict, model, preprocessor_meta):
    feature_names = preprocessor_meta["feature_names"]
    numerical_cols = preprocessor_meta["numerical_cols"]
    scaler = preprocessor_meta["scaler"]

    row = pd.DataFrame([input_dict])
    cat_cols = row.select_dtypes(include=["object"]).columns.tolist()
    row = pd.get_dummies(row, columns=cat_cols, drop_first=False)
    row = row.reindex(columns=feature_names, fill_value=0).astype(float)

    num_present = [c for c in numerical_cols if c in row.columns]
    if num_present:
        row[num_present] = scaler.transform(row[num_present])

    pred = int(model.predict(row)[0])
    proba = 0.0
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(row)[0, 1])
    return pred, proba

def predict_disease(input_dict, disease_bundle):
    feature_names = disease_bundle["feature_names"]
    numerical_cols = disease_bundle["numerical_cols"]
    scaler = disease_bundle["scaler"]
    le = disease_bundle["label_encoder"]
    model = disease_bundle["model"]

    row = pd.DataFrame([input_dict])
    cat_cols = row.select_dtypes(include=["object"]).columns.tolist()
    row = pd.get_dummies(row, columns=cat_cols, drop_first=False)
    row = row.reindex(columns=feature_names, fill_value=0).astype(float)

    num_present = [c for c in numerical_cols if c in row.columns]
    if num_present:
        row[num_present] = scaler.transform(row[num_present])

    proba = model.predict_proba(row)[0]
    classes = le.classes_
    order = np.argsort(proba)[::-1]
    best_label = str(classes[order[0]])
    return best_label

def main():
    print("==========================================================================================")
    print("ANOMALİ VE HASTALIK TUTARLILIK TESTİ")
    print("==========================================================================================")
    
    save_dir = "saved_models"
    metadata_path = os.path.join(save_dir, "metadata.joblib")
    preprocessor_path = os.path.join(save_dir, "preprocessor.joblib")
    disease_path = os.path.join(save_dir, "disease_classifier.joblib")
    
    if not os.path.exists(metadata_path) or not os.path.exists(disease_path):
        print("Hata: Gerekli model dosyaları bulunamadı. Lütfen train_models.py'yi çalıştırın.")
        return
        
    print("[1] Modeller ve Metadata yükleniyor...")
    metadata = joblib.load(metadata_path)
    preprocessor_meta = joblib.load(preprocessor_path)
    disease_bundle = joblib.load(disease_path)
    
    best_model_name = metadata.get("best_model", "XGBoost")
    safe_name = best_model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    model_path = os.path.join(save_dir, f"{safe_name}.joblib")
    model = joblib.load(model_path)
    
    print("[2] Veri Seti Yükleniyor...")
    loader = DataLoader(data_dir="archive")
    raw_df = loader.load()
    # Hastalık analizi için id ve leakage sütunlarını at, ama label'ları tut
    df_clean = loader.clean(raw_df)
    
    # Gerçek hastalığı bilinen (NaN olmayan) örneklerden seçelim
    if "disease_category" not in df_clean.columns:
        # Clean adımında disease_category düşmüşse raw'dan alalım
        df_clean["disease_category"] = raw_df["disease_category"]
        
    df_valid = df_clean.dropna(subset=["disease_category"])
        
    print(f"[3] Gerçek hastalığı bilinenlerden rastgele 15 örnek seçiliyor...\n")
    sample_indices = np.random.choice(len(df_valid), 15, replace=False)
    df_sample = df_valid.iloc[sample_indices]
    
    print(f"{'İndeks':<8} | {'Gerçek Durum':<15} | {'Anomali Tahmini':<18} | {'Hastalık Tahmini':<20} | {'Gerçek Hastalık':<20}")
    print("-" * 90)
    
    for idx, row in df_sample.iterrows():
        input_dict = row.to_dict()
        
        # Gerçek değerler
        true_anomaly = int(row.get("anomaly_label", -1))
        true_disease = str(row.get("disease_category", "Bilinmiyor"))
        
        # Tahminler
        pred_anom, prob_anom = predict_anomaly(input_dict, model, preprocessor_meta)
        pred_dx = predict_disease(input_dict, disease_bundle)
        
        # Formatlama
        true_anom_txt = "Anormal" if true_anomaly == 1 else "Normal"
        pred_anom_txt = f"Anormal (%{prob_anom*100:.1f})" if pred_anom == 1 else f"Normal (%{prob_anom*100:.1f})"
        
        print(f"{idx:<8} | {true_anom_txt:<15} | {pred_anom_txt:<18} | {pred_dx:<20} | {true_disease:<20}")
        
    print("\n==========================================================================================")
    print("TEST TAMAMLANDI")
    print("Bu script iki modelin (Anomali vs Hastalık) birbiriyle tutarlı olup olmadığını ölçer.")
    print("Not: Örneğin anomali modelinin 'Anormal' dediğine, hastalık modeli de hasta demiş mi görebilirsiniz.")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
