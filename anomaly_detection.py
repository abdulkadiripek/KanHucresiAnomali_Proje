"""
Kan Hücresi Anomali Tespiti — ML Çekirdek Kütüphanesi
=====================================================
Veri yükleme, ön işleme, model eğitimi, değerlendirme, görselleştirme ve SHAP.

Tüm ML mantığı bu dosyada — train_models.py ve app.py buradan import eder.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    average_precision_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid")

LEAKAGE_COLS = [
    "disease_category", "cell_type",
    "cytodiffusion_anomaly_score",
    "cytodiffusion_classification_confidence",
    "labeller_confidence_score",
]
ID_COLS = ["cell_id"]
TARGET = "anomaly_label"
MODEL_NAMES = ["XGBoost", "LightGBM", "Random Forest"]


# ============================================================================
# 1. VERİ
# ============================================================================
def load_data(data_dir: str, filename: str = "blood_cell_anomaly_detection.csv"):
    return pd.read_csv(os.path.join(data_dir, filename))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in LEAKAGE_COLS + ID_COLS if c in df.columns]
    return df.drop(columns=cols)


def get_summary(df: pd.DataFrame) -> dict:
    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "n_anomalies": int(df[TARGET].sum()) if TARGET in df.columns else 0,
        "anomaly_ratio": float(df[TARGET].mean()) if TARGET in df.columns else 0.0,
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


# ============================================================================
# 2. ÖN İŞLEME
# ============================================================================
def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """One-Hot + stratified split + StandardScaler (sadece sayısal sütunlar)."""
    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    X = pd.get_dummies(X, columns=cat_cols, drop_first=True).astype(float)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    num_present = [c for c in num_cols if c in X_tr.columns]
    X_tr[num_present] = scaler.fit_transform(X_tr[num_present])
    X_te[num_present] = scaler.transform(X_te[num_present])

    return {
        "X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te,
        "scaler": scaler,
        "feature_names": X_tr.columns.tolist(),
        "numerical_cols": num_present,
    }


# ============================================================================
# 3. MODELLER
# ============================================================================
def build_model(name: str, random_state: int = 42):
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=random_state,
            verbosity=0, nthread=1,
        )
    if name == "LightGBM":
        return LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, verbose=-1, n_jobs=1,
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300, min_samples_split=5, min_samples_leaf=2,
            random_state=random_state, n_jobs=-1,
        )
    raise ValueError(f"Bilinmeyen model: {name}")


def train_with_cv(model, X_train, y_train, X_test, n_folds: int = 5,
                  random_state: int = 42):
    """5-fold StratifiedKFold CV + final fit + test tahmini."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                scoring="f1", n_jobs=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return {
        "model": model, "y_pred": y_pred, "y_proba": y_proba,
        "cv_scores": cv_scores,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
    }


# ============================================================================
# 4. DEĞERLENDİRME
# ============================================================================
def evaluate(y_true, y_pred, y_proba=None, cv_mean=None, cv_std=None) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "confusion_mat": confusion_matrix(y_true, y_pred),
        "classification_rep": classification_report(
            y_true, y_pred, target_names=["Normal", "Anomali"]
        ),
        "cv_mean": cv_mean, "cv_std": cv_std,
    }
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        out["roc_auc"] = auc(fpr, tpr)
        out["pr_auc"] = average_precision_score(y_true, y_proba)
    return out


def comparison_table(eval_results: dict) -> pd.DataFrame:
    rows = []
    for name, r in eval_results.items():
        rows.append({
            "Model": name,
            "Accuracy": r["accuracy"], "Bal. Acc.": r["balanced_accuracy"],
            "Precision": r["precision"], "Recall": r["recall"],
            "F1-Score": r["f1"], "MCC": r["mcc"], "Kappa": r["kappa"],
            "ROC-AUC": r.get("roc_auc"), "PR-AUC": r.get("pr_auc"),
            "CV Mean (F1)": r.get("cv_mean"),
        })
    return pd.DataFrame(rows).set_index("Model").sort_values("F1-Score", ascending=False)


# ============================================================================
# 5. GÖRSELLEŞTİRME
# ============================================================================
def plot_class_balance(y: pd.Series):
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(counts.values, labels=["Normal", "Anomali"], autopct="%1.1f%%",
           colors=["#3b82f6", "#ef4444"], startangle=90,
           wedgeprops=dict(width=0.4))
    ax.set_title("Sınıf Dağılımı", fontsize=13, weight="bold")
    return fig


def plot_leakage_correlations(df: pd.DataFrame, leakage_cols: list):
    num_df = df.select_dtypes(include=["number"])
    if TARGET not in num_df.columns:
        return None
    corr = num_df.corr()[TARGET].drop(TARGET).abs().sort_values()
    colors = ["#ef4444" if c in leakage_cols else "#3b82f6" for c in corr.index]
    fig, ax = plt.subplots(figsize=(8, max(4, len(corr) * 0.3)))
    ax.barh(corr.index, corr.values, color=colors)
    ax.set_xlabel("|Pearson korelasyonu| (anomaly_label ile)")
    ax.set_title("Leakage Kanıtı — kırmızı: çıkarılan sütunlar", weight="bold")
    plt.tight_layout()
    return fig


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def plot_cohens_d_top(df: pd.DataFrame, top_n: int = 10):
    num = df.select_dtypes(include=["number"]).drop(columns=[TARGET], errors="ignore")
    g0 = df[df[TARGET] == 0]
    g1 = df[df[TARGET] == 1]
    d_vals = {c: abs(cohens_d(g1[c], g0[c])) for c in num.columns}
    s = pd.Series(d_vals).sort_values().tail(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, len(s) * 0.4)))
    ax.barh(s.index, s.values, color="#8b5cf6")
    for x, ls, lbl in [(0.2, "--", "küçük"), (0.5, "--", "orta"), (0.8, "--", "büyük")]:
        ax.axvline(x, color="gray", ls=ls, lw=0.7, alpha=0.6, label=f"{lbl} ({x})")
    ax.set_xlabel("|Cohen's d|")
    ax.set_title(f"Top {top_n} Ayırt Edici Özellik", weight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig, s.sort_values(ascending=False).index.tolist()


def plot_kde_grid(df: pd.DataFrame, features: list):
    if not features:
        return None
    n = len(features)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(11, 3.5 * rows))
    axes = np.array(axes).flatten()
    for i, feat in enumerate(features):
        ax = axes[i]
        for label, color in [(0, "#3b82f6"), (1, "#ef4444")]:
            data = df[df[TARGET] == label][feat].dropna()
            if len(data) > 1:
                sns.kdeplot(data, ax=ax, color=color, fill=True, alpha=0.4,
                            label="Normal" if label == 0 else "Anomali")
        ax.set_title(feat, fontsize=10)
        ax.legend(fontsize=8)
    for j in range(len(features), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    return fig


def plot_categorical_bias(df: pd.DataFrame, cat_cols: list):
    """Her kategorik sütun için anomali oranı barları."""
    cols = [c for c in cat_cols if c in df.columns]
    if not cols:
        return None
    baseline = df[TARGET].mean()
    n = len(cols)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.5 * rows))
    axes = np.array(axes).flatten()
    for i, c in enumerate(cols):
        ax = axes[i]
        rates = df.groupby(c)[TARGET].mean().sort_values()
        ax.bar(rates.index.astype(str), rates.values, color="#10b981")
        ax.axhline(baseline, color="red", ls="--", lw=1,
                   label=f"baseline {baseline:.1%}")
        ax.set_title(c, fontsize=10)
        ax.set_ylabel("Anomali oranı")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=7)
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    return fig


def plot_confusion(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Anomali"],
                yticklabels=["Normal", "Anomali"], cbar=False)
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
    ax.set_title(f"Confusion Matrix — {model_name}", weight="bold")
    plt.tight_layout()
    return fig


def plot_roc(roc_data: dict):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, (fpr, tpr, a) in roc_data.items():
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Eğrileri", weight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_pr(pr_data: dict):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, (prec, rec, ap) in pr_data.items():
        ax.plot(rec, prec, lw=2, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Eğrileri", weight="bold")
    ax.legend(loc="lower left")
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names: list, top_n: int = 15):
    if not hasattr(model, "feature_importances_"):
        return None
    s = pd.Series(model.feature_importances_, index=feature_names).sort_values().tail(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, len(s) * 0.35)))
    ax.barh(s.index, s.values, color="#10b981")
    ax.set_title(f"Özellik Önemleri (Top {top_n})", weight="bold")
    plt.tight_layout()
    return fig


def plot_metric_comparison(comp_df: pd.DataFrame):
    metrics = [m for m in ["F1-Score", "Accuracy", "Precision", "Recall", "ROC-AUC"]
               if m in comp_df.columns]
    fig, ax = plt.subplots(figsize=(9, 5))
    comp_df[metrics].plot(kind="bar", ax=ax, colormap="viridis", width=0.8)
    ax.set_ylabel("Skor"); ax.set_ylim(0, 1.05)
    ax.set_title("Model Karşılaştırması", weight="bold")
    ax.legend(loc="lower right", fontsize=8)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    return fig


# ============================================================================
# 6. SHAP
# ============================================================================
def compute_shap(model, X_sample: pd.DataFrame, X_background: pd.DataFrame = None):
    cls_name = type(model).__name__
    tree_models = ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier")
    if cls_name in tree_models:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        bg = X_background if X_background is not None else X_sample.head(50)
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        shap_values = explainer.shap_values(X_sample, silent=True)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.array(base).flatten()[-1])

    return {
        "shap_values": shap_values,
        "base_value": float(base),
        "X_sample": X_sample,
        "feature_names": list(X_sample.columns),
    }


def plot_shap_bar(result: dict, top_n: int = 15):
    mean_abs = np.abs(result["shap_values"]).mean(axis=0)
    s = pd.Series(mean_abs, index=result["feature_names"]).sort_values().tail(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, len(s) * 0.35)))
    ax.barh(s.index, s.values, color="#f59e0b")
    ax.set_xlabel("Ortalama |SHAP|")
    ax.set_title(f"Global Özellik Önemi — SHAP (Top {top_n})", weight="bold")
    plt.tight_layout()
    return fig


def plot_shap_summary(result: dict, top_n: int = 15):
    plt.figure(figsize=(9, max(5, top_n * 0.4)))
    shap.summary_plot(result["shap_values"], result["X_sample"],
                      max_display=top_n, show=False)
    return plt.gcf()


def plot_shap_waterfall(result: dict, idx: int, top_n: int = 12):
    shap_row = result["shap_values"][idx]
    x_row = result["X_sample"].iloc[idx]
    order = np.argsort(np.abs(shap_row))[-top_n:]
    feats = [result["feature_names"][i] for i in order]
    vals = shap_row[order]
    feat_vals = x_row.values[order]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.45)))
    colors = ["#ef4444" if v > 0 else "#3b82f6" for v in vals]
    ax.barh(range(len(feats)), vals, color=colors)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([f"{f} = {v:.3f}" for f, v in zip(feats, feat_vals)],
                       fontsize=9)
    ax.axvline(0, color="black", lw=0.5)
    total = shap_row.sum()
    ax.set_title(
        f"Lokal Açıklama #{idx} — base={result['base_value']:+.3f}, "
        f"toplam SHAP={total:+.3f}", weight="bold"
    )
    ax.set_xlabel("SHAP değeri")
    plt.tight_layout()
    return fig


# ============================================================================
# 7. TEK ÖRNEK TAHMİN
# ============================================================================
def predict_single(input_dict: dict, model, preprocessor: dict) -> tuple:
    feature_names = preprocessor["feature_names"]
    num_cols = preprocessor["numerical_cols"]
    scaler = preprocessor["scaler"]

    row = pd.DataFrame([input_dict])
    cat_cols = row.select_dtypes(include=["object"]).columns.tolist()
    row = pd.get_dummies(row, columns=cat_cols, drop_first=True)
    row = row.reindex(columns=feature_names, fill_value=0).astype(float)

    num_present = [c for c in num_cols if c in row.columns]
    if num_present:
        row[num_present] = scaler.transform(row[num_present])

    pred = int(model.predict(row)[0])
    proba = float(model.predict_proba(row)[0, 1]) if hasattr(model, "predict_proba") else None
    return pred, proba


def predict_disease(input_dict: dict, bundle: dict) -> tuple:
    row = pd.DataFrame([input_dict])
    cat_cols = row.select_dtypes(include=["object"]).columns.tolist()
    row = pd.get_dummies(row, columns=cat_cols, drop_first=True)
    row = row.reindex(columns=bundle["feature_names"], fill_value=0).astype(float)

    num_present = [c for c in bundle["numerical_cols"] if c in row.columns]
    if num_present:
        row[num_present] = bundle["scaler"].transform(row[num_present])

    proba = bundle["model"].predict_proba(row)[0]
    classes = bundle["label_encoder"].classes_
    order = np.argsort(proba)[::-1]
    top3 = [(str(classes[i]), float(proba[i])) for i in order[:3]]

    # Normal kategorilerin toplam olasılığı
    normal_proba = sum(
        float(proba[i]) for i, c in enumerate(classes)
        if str(c).lower().startswith("normal")
    )
    return str(classes[order[0]]), top3, normal_proba


# ============================================================================
# 8. HASTALIK SINIFLANDIRICI (multi-class)
# ============================================================================
def train_disease_classifier(raw_df: pd.DataFrame, random_state: int = 42) -> dict:
    drop_cols = [c for c in LEAKAGE_COLS + ID_COLS
                 if c in raw_df.columns and c != "disease_category"]
    drop_cols.append(TARGET)
    df = raw_df.drop(columns=[c for c in drop_cols if c in raw_df.columns])
    df = df.dropna(subset=["disease_category"])

    y_raw = df["disease_category"]
    X = df.drop(columns=["disease_category"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    orig_num = X.select_dtypes(include=["number"]).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True).astype(float)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    num_present = [c for c in orig_num if c in X_tr.columns]
    X_tr[num_present] = scaler.fit_transform(X_tr[num_present])
    X_te[num_present] = scaler.transform(X_te[num_present])

    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=len(le.classes_),
        eval_metric="mlogloss", random_state=random_state,
        verbosity=0, nthread=1,
    )
    model.fit(X_tr, y_tr)
    acc = float((model.predict(X_te) == y_te).mean())

    return {
        "model": model, "scaler": scaler,
        "feature_names": X_tr.columns.tolist(),
        "numerical_cols": num_present,
        "label_encoder": le,
        "classes": list(le.classes_),
        "accuracy": acc,
    }
