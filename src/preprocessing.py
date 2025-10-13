# define preprocessing functions -- which should be defined for each type of model to be used. 
# windowing functions, capping, outlier detection.
# function to window looking backwards, selecting the dataframe, the date feature, 
# and list of features to be considered.

import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Dict, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin


class AutoPreprocessor(BaseEstimator, TransformerMixin):
    """
    A preprocessor that automates feature detection, cleaning, imputation,
    and encoding based on the logic from ModelDataProfiler.

    It identifies and handles:
    - Numeric-like string columns.
    - Date columns.
    - High-cardinality (ID-like) columns.
    - Missing values (imputing and adding indicator columns).
    - One-hot encoding for categorical features.
    """

    def __init__(self, high_cardinality_threshold: float = 0.95, verbose: bool = True):
        """
        Initializes the preprocessor.

        Args:
            high_cardinality_threshold (float): The ratio of unique values to total
                values above which a feature is considered an ID.
            verbose (bool): If True, prints information during fitting.
        """
        self.high_cardinality_threshold = high_cardinality_threshold
        self.verbose = verbose
        self.initial_numerical_features_: List[str] = []
        self.initial_categorical_features_: List[str] = []
        self.date_features_: List[str] = []
        self.id_features_: List[str] = []
        self.imputation_values_: Dict[str, Any] = {}
        self.final_features_: List[str] = []
        self.categorical_to_encode_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """
        Learns the transformations to be applied to the data.

        Args:
            X (pd.DataFrame): The input dataframe.
        """
        if self.verbose: print("--- Starting AutoPreprocessor fit ---")
        df = X.copy()

        # 1. Initial feature type detection
        self.initial_numerical_features_ = df.select_dtypes(include=np.number).columns.tolist()
        self.initial_categorical_features_ = df.select_dtypes(exclude=np.number).columns.tolist()

        # 2. Detect and re-assign feature types
        self._detect_feature_types(df)

        # 3. Learn imputation values and identify columns needing indicators
        numerical_to_impute = [col for col in self.initial_numerical_features_ if col not in self.id_features_]
        categorical_to_impute = [col for col in self.initial_categorical_features_ if col not in self.id_features_ and col not in self.date_features_]

        for col in numerical_to_impute:
            if df[col].isnull().any():
                self.imputation_values_[col] = df[col].median()

        for col in categorical_to_impute:
            if df[col].isnull().any():
                self.imputation_values_[col] = df[col].mode()[0]

        self.categorical_to_encode_ = categorical_to_impute

        if self.verbose: print("--- AutoPreprocessor fit complete ---")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the learned transformations to the data.

        Args:
            X (pd.DataFrame): The input dataframe to transform.

        Returns:
            pd.DataFrame: The transformed dataframe.
        """
        if self.verbose: print("--- Starting AutoPreprocessor transform ---")
        df = X.copy()

        # 1. Drop identified date and ID features
        features_to_drop = self.date_features_ + self.id_features_
        df.drop(columns=[col for col in features_to_drop if col in df.columns], inplace=True)
        if self.verbose: print(f"Dropped columns: {features_to_drop}")

        # 2. Create missing value indicators and impute
        for col, value in self.imputation_values_.items():
            if col in df.columns:
                indicator_col = f"{col}_was_missing"
                if df[col].isnull().any():
                    df[indicator_col] = df[col].isnull().astype(int)
                    df[col].fillna(value, inplace=True)
                    if self.verbose: print(f"Imputed '{col}' and created '{indicator_col}'")
                else: # Ensure column exists even if no missing values in this split
                    df[indicator_col] = 0

        # 3. One-hot encode categorical features
        if self.categorical_to_encode_:
            if self.verbose: print(f"One-hot encoding: {self.categorical_to_encode_}")
            df = pd.get_dummies(df, columns=self.categorical_to_encode_, drop_first=True)

        # Store final feature list on first transform
        if not self.final_features_:
            self.final_features_ = df.columns.tolist()

        # Align columns with what was seen during fit
        df = df.reindex(columns=self.final_features_, fill_value=0)

        if self.verbose: print(f"--- Transform complete. Final shape: {df.shape} ---")
        return df

    def _detect_feature_types(self, df: pd.DataFrame):
        """Helper to detect and re-assign feature types."""
        
        # Check for numeric-like categorical features
        for col in self.initial_categorical_features_[:]:
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                converted_series = pd.to_numeric(df[col], errors='coerce')
                if converted_series.isnull().sum() == df[col].isnull().sum():
                    if self.verbose: print(f"'{col}' detected as numeric-like string. Re-classifying.")
                    self.initial_categorical_features_.remove(col)
                    self.initial_numerical_features_.append(col)

        # Check for date-like features
        for col in self.initial_categorical_features_[:]:
            if df[col].isnull().all(): continue
            try:
                converted_series = pd.to_datetime(df[col], errors='coerce')
                if converted_series.notna().sum() / df[col].notna().sum() > 0.95:
                    if self.verbose: print(f"'{col}' detected as date feature.")
                    self.date_features_.append(col)
                    self.initial_categorical_features_.remove(col)
            except Exception:
                continue

        # Check for high-cardinality (ID) features
        for col in self.initial_categorical_features_[:]:
            if df[col].nunique() / len(df) > self.high_cardinality_threshold:
                if self.verbose: print(f"'{col}' detected as high-cardinality categorical (ID).")
                self.id_features_.append(col)
                self.initial_categorical_features_.remove(col)

        for col in self.initial_numerical_features_[:]:
            is_high_cardinality = df[col].nunique() / len(df) > self.high_cardinality_threshold
            if is_high_cardinality:
                # Heuristic: integer-like, low skew
                is_float_like = (df[col].dropna() % 1 != 0).any()
                if not is_float_like:
                    if self.verbose: print(f"'{col}' detected as high-cardinality numerical (ID).")
                    self.id_features_.append(col)
                    self.initial_numerical_features_.remove(col)
