"""
DataLoader — Veri yükleme, doğrulama ve temizleme işlemleri.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional


class DataLoader:
    """Kan hücresi veri setini yükler, doğrular ve temizler.

    Attributes:
        data_dir: Veri dosyalarının bulunduğu dizin.
        target_col: Hedef değişken sütun adı.
        leakage_cols: Data leakage riski taşıyan sütunlar.
        id_cols: Kimlik sütunları (modellemeye katkısı yok).
    """

    DEFAULT_LEAKAGE_COLS = [
        "disease_category",
        "cell_type",
        "cytodiffusion_anomaly_score",
        "cytodiffusion_classification_confidence",
        "labeller_confidence_score",
    ]
    DEFAULT_ID_COLS = ["cell_id"]

    def __init__(
        self,
        data_dir: str,
        target_col: str = "anomaly_label",
        leakage_cols: Optional[list] = None,
        id_cols: Optional[list] = None,
    ):
        self.data_dir = data_dir
        self.target_col = target_col
        self.leakage_cols = leakage_cols or self.DEFAULT_LEAKAGE_COLS
        self.id_cols = id_cols or self.DEFAULT_ID_COLS
        self._raw_df: Optional[pd.DataFrame] = None
        self._clean_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, filename: str = "blood_cell_anomaly_detection.csv") -> pd.DataFrame:
        """CSV dosyasını yükle ve ham veriyi döndür."""
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Veri dosyası bulunamadı: {path}")
        self._raw_df = pd.read_csv(path)
        return self._raw_df.copy()

    def clean(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Leakage ve kimlik sütunlarını kaldır.

        Args:
            df: Temizlenecek DataFrame. None ise son yüklenen veri kullanılır.

        Returns:
            Temizlenmiş DataFrame.
        """
        if df is None:
            if self._raw_df is None:
                raise ValueError("Önce load() ile veri yükleyin.")
            df = self._raw_df.copy()

        cols_to_drop = [c for c in self.leakage_cols + self.id_cols if c in df.columns]
        self._clean_df = df.drop(columns=cols_to_drop)
        return self._clean_df.copy()

    def get_summary(self, df: Optional[pd.DataFrame] = None) -> dict:
        """Veri seti hakkında özet istatistikler döndür."""
        if df is None:
            df = self._raw_df if self._raw_df is not None else pd.DataFrame()

        summary = {
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "n_anomalies": int(df[self.target_col].sum()) if self.target_col in df.columns else 0,
            "anomaly_ratio": float(df[self.target_col].mean()) if self.target_col in df.columns else 0.0,
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "dtypes": df.dtypes.value_counts().to_dict(),
        }
        return summary

    def get_column_info(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Her sütun için detaylı bilgi tablosu döndür."""
        if df is None:
            df = self._raw_df if self._raw_df is not None else pd.DataFrame()

        info = pd.DataFrame({
            "Sütun": df.columns,
            "Veri Tipi": df.dtypes.astype(str).values,
            "Eksik Değer": df.isnull().sum().values,
            "Eksik Oranı (%)": (df.isnull().sum().values / len(df) * 100).round(2),
            "Benzersiz Değer": df.nunique().values,
        })
        return info

    @property
    def raw_data(self) -> Optional[pd.DataFrame]:
        return self._raw_df.copy() if self._raw_df is not None else None

    @property
    def clean_data(self) -> Optional[pd.DataFrame]:
        return self._clean_df.copy() if self._clean_df is not None else None
