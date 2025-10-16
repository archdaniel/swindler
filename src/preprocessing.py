# define preprocessing functions -- which should be defined for each type of model to be used. 
# windowing functions, capping, outlier detection.
# function to window looking backwards, selecting the dataframe, the date feature, 
# and list of features to be considered.

import pandas as pd
import numpy as np
from pandas.core.api import isnull
import re
from typing import List, Tuple, Dict, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher

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

    def __init__(self, id_threshold: float = 0.95,
                 cardinality_threshold: int = 50,
                 high_cardinality_strategy: str = 'grouping',
                 n_hashing_features: int = 20,
                 rare_category_threshold: float = 0.01,
                 verbose: bool = True):
        """
        Initializes the preprocessor.

        Args:
            id_threshold (float): The ratio of unique values to total
                values above which a feature is considered an ID.
            cardinality_threshold (int): The number of unique values in a categorical
                feature above which it is considered "high-cardinality".
            high_cardinality_strategy (str): Method for high-cardinality features.
                'grouping': Bins rare categories into a single 'rare' group.
                'hashing': Uses the Feature Hashing trick.
            n_hashing_features (int): Number of output features for the hasher if
                `high_cardinality_strategy` is 'hashing'.
            rare_category_threshold (float): The minimum frequency for a category to be 
                kept as a separate feature. Categories below this are grouped.
            verbose (bool): If True, prints information during fitting.
        """
        self.id_threshold = id_threshold
        self.cardinality_threshold = cardinality_threshold
        self.high_cardinality_strategy = high_cardinality_strategy
        self.n_hashing_features = n_hashing_features
        self.rare_category_threshold = rare_category_threshold
        self.verbose = verbose
        self.initial_numerical_features_: List[str] = []
        self.initial_categorical_features_: List[str] = []
        self.low_cardinality_features_: List[str] = []
        self.high_cardinality_features_: List[str] = []
        self.date_features_: List[str] = []
        self.id_features_: List[str] = []
        self.imputation_values_: Dict[str, Any] = {}
        self.final_features_: List[str] = []
        self.frequent_categories_: Dict[str, List[str]] = {}
        self.hasher_: FeatureHasher = None
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
        numerical_to_impute = [col for col in self.initial_numerical_features_ 
                               if col not in self.id_features_]
        categorical_to_impute = [col for col in self.initial_categorical_features_ 
                                 if col not in self.id_features_ and col not in self.date_features_]

        for col in numerical_to_impute:
            if df[col].isnull().any():
                self.imputation_values_[col] = df[col].median()

        for col in categorical_to_impute:
            if df[col].isnull().any():
                self.imputation_values_[col] = df[col].mode()[0]

        # 4. Separate categorical features and prepare for the chosen strategy
        for col in categorical_to_impute:
            if df[col].nunique() > self.cardinality_threshold:
                self.high_cardinality_features_.append(col)
            else:
                self.low_cardinality_features_.append(col)

        if self.high_cardinality_strategy == 'grouping':
            if self.verbose: print("Fit strategy: Grouping rare categories.")
            for col in self.high_cardinality_features_:
                counts = df[col].value_counts(normalize=True)
                self.frequent_categories_[col] = counts[counts >= self.rare_category_threshold].index.tolist()
                if self.verbose: print(f"'{col}' is high-cardinality. Found {len(self.frequent_categories_[col])} frequent categories.")
            self.categorical_to_encode_ = self.low_cardinality_features_ + self.high_cardinality_features_

        elif self.high_cardinality_strategy == 'hashing':
            if self.verbose: print(f"Fit strategy: Feature Hashing to {self.n_hashing_features} features.")
            if self.high_cardinality_features_:
                self.hasher_ = FeatureHasher(n_features=self.n_hashing_features, input_type='dict')
            self.categorical_to_encode_ = self.low_cardinality_features_
        else:
            raise ValueError(f"Unknown high_cardinality_strategy: '{self.high_cardinality_strategy}'")

        if self.verbose: print(f"Low cardinality features to OHE: {self.low_cardinality_features_}")

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

        processed_parts = []

        # 2. Create missing value indicators and impute
        for col, value in self.imputation_values_.items():
            if col in df.columns:
                is_missing = df[col].isnull()
                if is_missing.any():
                    df[col].fillna(value, inplace=True)
                    if self.verbose: print(f"Imputed '{col}' with '{value}'")
                indicator_col = f"{col}_was_missing"
                df[indicator_col] = is_missing.astype(int)
                if self.verbose and is_missing.any(): print(f"Created '{indicator_col}'")

        # Part 1: Numerical and missing indicator columns
        numeric_cols = [col for col in self.initial_numerical_features_ if col in df.columns]
        missing_indicators = [col for col in df.columns if col.endswith('_was_missing')]
        processed_parts.append(df[numeric_cols + missing_indicators].reset_index(drop=True))

        # Part 2: Low-cardinality features (always one-hot encoded)
        if self.categorical_to_encode_:
            if self.verbose: print(f"One-hot encoding low-cardinality: {self.categorical_to_encode_}")
            df_low_card = pd.get_dummies(df[self.categorical_to_encode_], columns=self.categorical_to_encode_, drop_first=True)
            processed_parts.append(df_low_card.reset_index(drop=True))

        # Part 3: High-cardinality features (apply chosen strategy)
        if self.high_cardinality_features_:
            df_high_card = df[self.high_cardinality_features_]

            if self.high_cardinality_strategy == 'grouping':
                if self.verbose: print("Transform: Grouping rare categories.")
                for col in self.high_cardinality_features_:
                    frequent_cats = self.frequent_categories_.get(col, [])
                    df_high_card[col] = df_high_card[col].where(df_high_card[col].isin(frequent_cats), 'rare_category')
                df_high_card_processed = pd.get_dummies(df_high_card, columns=self.high_cardinality_features_, drop_first=True)
                processed_parts.append(df_high_card_processed.reset_index(drop=True))

            elif self.high_cardinality_strategy == 'hashing' and self.hasher_:
                if self.verbose: print("Transform: Applying Feature Hashing.")
                dict_rows = df_high_card.to_dict(orient='records')
                hashed_features = self.hasher_.transform(dict_rows)
                df_hashed = pd.DataFrame(
                    hashed_features.toarray(),
                    columns=[f'hash_{i}' for i in range(self.n_hashing_features)]
                )
                processed_parts.append(df_hashed.reset_index(drop=True))

        # Combine all processed parts
        df_transformed = pd.concat(processed_parts, axis=1)

        # Store final feature list on first transform
        if not self.final_features_:
            self.final_features_ = df_transformed.columns.tolist()

        # Align columns with what was seen during fit
        df_transformed = df_transformed.reindex(columns=self.final_features_, fill_value=0)

        if self.verbose: print(f"--- Transform complete. Final shape: {df_transformed.shape} ---")
        return df_transformed

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
            if df[col].nunique() / len(df) > self.id_threshold:
                if self.verbose: print(f"'{col}' detected as high-cardinality categorical (ID).")
                self.id_features_.append(col)
                self.initial_categorical_features_.remove(col)

        for col in self.initial_numerical_features_[:]:
            is_high_cardinality = df[col].nunique() / len(df) > self.id_threshold
            if is_high_cardinality:
                # Heuristic: integer-like, low skew
                is_float_like = (df[col].dropna() % 1 != 0).any()
                if not is_float_like:
                    if self.verbose: print(f"'{col}' detected as high-cardinality numerical (ID).")
                    self.id_features_.append(col)
                    self.initial_numerical_features_.remove(col)
