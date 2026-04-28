"""
ModelFactory — Model oluşturma, eğitim, cross-validation ve hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import joblib
import os
from dataclasses import dataclass, field
from typing import Optional, Any

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.base import clone
from sklearn.metrics import f1_score as _f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


@dataclass
class TrainResult:
    """Bir modelin eğitim sonuçları."""
    model_name: str
    model: Any
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]
    cv_scores: Optional[np.ndarray]
    cv_mean: Optional[float]
    cv_std: Optional[float]
    best_params: Optional[dict] = None


class ModelFactory:
    """Model oluşturma ve eğitim fabrikası.

    Desteklenen modeller:
      - XGBoost
      - LightGBM
      - Random Forest
      - Extra Trees
      - Gradient Boosting
      - AdaBoost
      - Logistic Regression
      - SVM (RBF kernel)

    Özellikler:
      - StratifiedKFold cross-validation
      - RandomizedSearchCV hyperparameter tuning
      - Ensemble (Voting Classifier)
      - Model kaydetme/yükleme (joblib)
    """

    # Varsayılan hiperparametre arama uzayları
    PARAM_GRIDS = {
        "XGBoost": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.1, 0.2],
        },
        "LightGBM": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [-1, 5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "num_leaves": [31, 50, 70, 100],
            "min_child_samples": [10, 20, 30],
        },
        "Random Forest": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        },
        "Extra Trees": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
        },
    }

    def __init__(self, random_state: int = 42, n_cv_folds: int = 5):
        self.random_state = random_state
        self.n_cv_folds = n_cv_folds
        self._cv = StratifiedKFold(
            n_splits=n_cv_folds, shuffle=True, random_state=random_state
        )
        self._trained_models: dict[str, TrainResult] = {}

    # ------------------------------------------------------------------
    # Model Oluşturma
    # ------------------------------------------------------------------

    def create_model(self, name: str, params: Optional[dict] = None) -> Any:
        """Bir ML modeli oluştur.

        Args:
            name: Model adı (örn. "XGBoost", "Random Forest").
            params: Hiperparametre dict'i. None ise varsayılanlar kullanılır.

        Returns:
            sklearn-uyumlu model nesnesi.
        """
        params = params or {}

        builders = {
            "XGBoost": self._build_xgboost,
            "LightGBM": self._build_lightgbm,
            "Random Forest": self._build_random_forest,
            "Extra Trees": self._build_extra_trees,
            "Gradient Boosting": self._build_gradient_boosting,
            "AdaBoost": self._build_adaboost,
            "Logistic Regression": self._build_logistic_regression,
            "SVM": self._build_svm,
        }

        builder = builders.get(name)
        if builder is None:
            raise ValueError(
                f"Bilinmeyen model: '{name}'. "
                f"Desteklenen modeller: {list(builders.keys())}"
            )
        return builder(params)

    # ------------------------------------------------------------------
    # Eğitim
    # ------------------------------------------------------------------

    def train(
        self,
        name: str,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        run_cv: bool = True,
    ) -> TrainResult:
        """Modeli eğit, cross-validasyon çalıştır ve test tahmini yap.

        Args:
            name: Model adı.
            model: sklearn-uyumlu model.
            X_train, y_train: Eğitim verileri.
            X_test, y_test: Test verileri.
            run_cv: Cross-validation yapılsın mı?

        Returns:
            TrainResult dataclass.
        """
        # Cross-validation — manuel döngü (OpenMP/joblib deadlock önlemi)
        cv_scores = None
        cv_mean = None
        cv_std = None
        if run_cv:
            cv_scores = self._manual_cv(model, X_train, y_train)
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())

        # Fit model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)

        result = TrainResult(
            model_name=name,
            model=model,
            y_pred=y_pred,
            y_proba=y_proba,
            cv_scores=cv_scores,
            cv_mean=cv_mean,
            cv_std=cv_std,
        )
        self._trained_models[name] = result
        return result

    # ------------------------------------------------------------------
    # Hyperparameter Tuning
    # ------------------------------------------------------------------

    def tune_hyperparameters(
        self,
        name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        n_iter: int = 30,
        scoring: str = "f1",
        param_grid: Optional[dict] = None,
    ) -> TrainResult:
        """RandomizedSearchCV ile hiperparametre optimizasyonu.

        Args:
            name: Model adı.
            X_train, y_train: Eğitim verileri.
            X_test: Test verileri.
            n_iter: Deneme sayısı.
            scoring: Optimizasyon metriği.
            param_grid: Arama uzayı. None ise varsayılan kullanılır.

        Returns:
            TrainResult (en iyi parametrelerle eğitilmiş model).
        """
        base_model = self.create_model(name)
        grid = param_grid or self.PARAM_GRIDS.get(name, {})

        if not grid:
            raise ValueError(f"'{name}' için parametre arama uzayı tanımlı değil.")

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=grid,
            n_iter=n_iter,
            cv=self._cv,
            scoring=scoring,
            random_state=self.random_state,
            n_jobs=1,
            verbose=0,
            pre_dispatch=1,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = None
        if hasattr(best_model, "predict_proba"):
            y_proba = best_model.predict_proba(X_test)[:, 1]

        result = TrainResult(
            model_name=f"{name} (Tuned)",
            model=best_model,
            y_pred=y_pred,
            y_proba=y_proba,
            cv_scores=None,
            cv_mean=float(search.best_score_),
            cv_std=None,
            best_params=search.best_params_,
        )
        self._trained_models[f"{name} (Tuned)"] = result
        return result

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------

    def create_ensemble(
        self,
        model_names: list,
        params_dict: Optional[dict] = None,
        voting: str = "soft",
    ) -> VotingClassifier:
        """Birden fazla modeli birleştiren VotingClassifier oluştur.

        Args:
            model_names: Modeller listesi.
            params_dict: Her model için hiperparametre dict'i.
            voting: "soft" (olasılık) veya "hard" (oy çoğunluğu).

        Returns:
            VotingClassifier.
        """
        params_dict = params_dict or {}
        estimators = []
        for name in model_names:
            model = self.create_model(name, params_dict.get(name, {}))
            estimators.append((name.lower().replace(" ", "_"), model))

        return VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)

    # ------------------------------------------------------------------
    # Model Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_model(model: Any, filepath: str) -> str:
        """Modeli joblib ile kaydet."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        return filepath

    @staticmethod
    def load_model(filepath: str) -> Any:
        """Kaydedilmiş modeli yükle."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {filepath}")
        return joblib.load(filepath)

    # ------------------------------------------------------------------
    @property
    def trained_models(self) -> dict:
        return dict(self._trained_models)

    # ------------------------------------------------------------------
    # Manuel CV (deadlock-safe)
    # ------------------------------------------------------------------

    def _manual_cv(self, model, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Saf Python döngüsüyle StratifiedKFold CV.

        joblib/multiprocessing kullanmaz → XGBoost ve LightGBM'in
        iç OpenMP thread havuzuyla deadlock oluşturmaz.
        """
        scores = []
        X_arr = X.values
        y_arr = y.values
        cols = X.columns.tolist()

        for fold_idx, (train_idx, val_idx) in enumerate(self._cv.split(X_arr, y_arr)):
            X_tr = pd.DataFrame(X_arr[train_idx], columns=cols)
            X_val = pd.DataFrame(X_arr[val_idx], columns=cols)
            y_tr = pd.Series(y_arr[train_idx])
            y_val = pd.Series(y_arr[val_idx])

            fold_model = clone(model)
            fold_model.fit(X_tr, y_tr)
            y_pred = fold_model.predict(X_val)
            scores.append(_f1_score(y_val, y_pred, zero_division=0))

        return np.array(scores)

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _build_xgboost(self, params: dict) -> XGBClassifier:
        defaults = {
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "eval_metric": "logloss", "random_state": self.random_state,
            "verbosity": 0,
            "nthread": 1,        # OpenMP deadlock önlemi
        }
        defaults.update(params)
        return XGBClassifier(**defaults)

    def _build_lightgbm(self, params: dict) -> LGBMClassifier:
        defaults = {
            "n_estimators": 300, "max_depth": -1, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "random_state": self.random_state, "verbose": -1,
            "n_jobs": 1,         # OpenMP deadlock önlemi
            "num_threads": 1,
        }
        defaults.update(params)
        return LGBMClassifier(**defaults)

    def _build_random_forest(self, params: dict) -> RandomForestClassifier:
        defaults = {
            "n_estimators": 300, "max_depth": None,
            "min_samples_split": 5, "min_samples_leaf": 2,
            "random_state": self.random_state, "n_jobs": -1,
        }
        defaults.update(params)
        return RandomForestClassifier(**defaults)

    def _build_extra_trees(self, params: dict) -> ExtraTreesClassifier:
        defaults = {
            "n_estimators": 300, "max_depth": None,
            "min_samples_split": 5, "min_samples_leaf": 2,
            "random_state": self.random_state, "n_jobs": -1,
        }
        defaults.update(params)
        return ExtraTreesClassifier(**defaults)

    def _build_gradient_boosting(self, params: dict) -> GradientBoostingClassifier:
        defaults = {
            "n_estimators": 200, "max_depth": 5, "learning_rate": 0.1,
            "subsample": 0.8, "random_state": self.random_state,
        }
        defaults.update(params)
        return GradientBoostingClassifier(**defaults)

    def _build_adaboost(self, params: dict) -> AdaBoostClassifier:
        defaults = {
            "n_estimators": 200, "learning_rate": 0.1,
            "random_state": self.random_state,
        }
        defaults.update(params)
        return AdaBoostClassifier(**defaults)

    def _build_logistic_regression(self, params: dict) -> LogisticRegression:
        defaults = {
            "max_iter": 1000, "random_state": self.random_state,
            "n_jobs": -1, "solver": "lbfgs",
        }
        defaults.update(params)
        return LogisticRegression(**defaults)

    def _build_svm(self, params: dict) -> SVC:
        defaults = {
            "kernel": "rbf", "probability": True,
            "random_state": self.random_state,
        }
        defaults.update(params)
        return SVC(**defaults)
