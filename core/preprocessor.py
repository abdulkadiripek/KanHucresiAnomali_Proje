"""
Preprocessor — Veri ön işleme: encoding, scaling, split, SMOTE.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold


@dataclass
class SplitResult:
    """Train/test split sonuçları."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list
    numerical_cols: list
    categorical_cols: list
    scaler: StandardScaler


class Preprocessor:
    """Veri ön işleme pipeline — encoding, scaling, split, feature selection.

    Sektörel best-practice'lere uygun:
      - Stratified split
      - Scaler sadece train üzerinde fit edilir
      - SMOTE opsiyonel (sadece train setine uygulanır)
      - Feature selection (mutual information, variance threshold)
    """

    def __init__(
        self,
        target_col: str = "anomaly_label",
        test_size: float = 0.20,
        random_state: int = 42,
        apply_smote: bool = False,
        variance_threshold: Optional[float] = None,
    ):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.apply_smote = apply_smote
        self.variance_threshold = variance_threshold
        self._scaler = StandardScaler()
        self._split_result: Optional[SplitResult] = None

    # ------------------------------------------------------------------
    def process(self, df: pd.DataFrame) -> SplitResult:
        """Tam ön işleme pipeline çalıştır.

        1. Hedef/özellik ayırma
        2. Kategorik/sayısal sütun tespiti
        3. One-Hot Encoding
        4. Variance threshold (opsiyonel)
        5. Train/Test split (stratified)
        6. StandardScaler (fit: train, transform: test)
        7. SMOTE (opsiyonel, sadece train)

        Returns:
            SplitResult dataclass.
        """
        y = df[self.target_col]
        X = df.drop(columns=[self.target_col])

        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()

        # One-Hot Encoding and ensure all feature columns are numeric/float for XGBoost
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        # Avoid pandas boolean/category type errors in XGBoost
        X = X.astype(float)

        # Variance Threshold
        if self.variance_threshold is not None:
            selector = VarianceThreshold(threshold=self.variance_threshold)
            mask = selector.get_support()
            X = X.loc[:, mask]

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # Scale numerical columns
        num_cols_present = [c for c in numerical_cols if c in X_train.columns]
        self._scaler = StandardScaler()
        X_train[num_cols_present] = self._scaler.fit_transform(X_train[num_cols_present])
        X_test[num_cols_present] = self._scaler.transform(X_test[num_cols_present])

        # SMOTE (sadece train setine)
        if self.apply_smote:
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=self.random_state)
                X_train_np, y_train_np = smote.fit_resample(X_train, y_train)
                X_train = pd.DataFrame(X_train_np, columns=X_train.columns)
                y_train = pd.Series(y_train_np, name=self.target_col)
            except ImportError:
                import warnings
                warnings.warn(
                    "imblearn paketi yüklü değil, SMOTE uygulanamadı. "
                    "Yüklemek için: pip install imbalanced-learn"
                )

        feature_names = X_train.columns.tolist()

        self._split_result = SplitResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            numerical_cols=num_cols_present,
            categorical_cols=categorical_cols,
            scaler=self._scaler,
        )
        return self._split_result

    # ------------------------------------------------------------------
    def compute_feature_importance_mi(
        self, X: pd.DataFrame, y: pd.Series, top_n: int = 20
    ) -> pd.DataFrame:
        """Mutual Information ile feature importance hesapla.

        Args:
            X: Özellik matrisi (scaled).
            y: Hedef değişken.
            top_n: Döndürülecek en önemli özellik sayısı.

        Returns:
            Sıralı feature importance DataFrame.
        """
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        mi_df = pd.DataFrame({
            "Feature": X.columns,
            "MI_Score": mi_scores,
        }).sort_values("MI_Score", ascending=False)
        return mi_df.head(top_n).reset_index(drop=True)

    @property
    def split_result(self) -> Optional[SplitResult]:
        return self._split_result
