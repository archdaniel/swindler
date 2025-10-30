# define preprocessing functions -- which should be defined for each type of model to be used. 
# windowing functions, capping, outlier detection.
# function to window looking backwards, selecting the dataframe, the date feature, 
# and list of features to be considered.

#  url=https://github.com/archdaniel/swindler/blob/main/src/preprocessing.py
# src/preprocessing.py
import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Dict, Any, Union, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler, LabelEncoder

class AutoPreprocessor(BaseEstimator, TransformerMixin):
    """
    A preprocessor that automates feature detection, cleaning, imputation,
    and encoding. This is mostly backwards-compatible with the file you had,
    but exposes hooks so a model-aware wrapper can configure behaviour.
    """
    def __init__(self, id_threshold: float = 0.95,
                 cardinality_threshold: int = 50,
                 high_cardinality_strategy: str = 'grouping',
                 n_hashing_features: int = 20,
                 rare_category_threshold: float = 0.01,
                 verbose: bool = True):
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
        self.hasher_: Optional[FeatureHasher] = None
        self.categorical_to_encode_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        if self.verbose: print("--- Starting AutoPreprocessor fit ---")
        df = X.copy()
        # initial detection
        self.initial_numerical_features_ = df.select_dtypes(include=np.number).columns.tolist()
        self.initial_categorical_features_ = df.select_dtypes(exclude=np.number).columns.tolist()

        # detect numeric-like stored as strings, dates and ids
        self._detect_feature_types(df)

        # prepare imputation map
        numerical_to_impute = [col for col in self.initial_numerical_features_ if col not in self.id_features_]
        categorical_to_impute = [col for col in self.initial_categorical_features_
                                 if col not in self.id_features_ and col not in self.date_features_]

        for col in numerical_to_impute:
            if df[col].isnull().any():
                self.imputation_values_[col] = df[col].median()

        for col in categorical_to_impute:
            if df[col].isnull().any():
                mode = df[col].mode(dropna=True)
                self.imputation_values_[col] = mode.iloc[0] if not mode.empty else "missing_category"

        # split into low/high cardinalities
        for col in categorical_to_impute:
            if df[col].nunique(dropna=False) > self.cardinality_threshold:
                self.high_cardinality_features_.append(col)
            else:
                self.low_cardinality_features_.append(col)

        # prepare hashing if requested
        if self.high_cardinality_strategy == 'grouping':
            for col in self.high_cardinality_features_:
                counts = df[col].value_counts(normalize=True)
                self.frequent_categories_[col] = counts[counts >= self.rare_category_threshold].index.tolist()
            self.categorical_to_encode_ = self.low_cardinality_features_ + self.high_cardinality_features_
        elif self.high_cardinality_strategy == 'hashing':
            if self.high_cardinality_features_:
                self.hasher_ = FeatureHasher(n_features=self.n_hashing_features, input_type='dict')
            # for hashing, we will one-hot/label-encode low-cardinals if needed
            self.categorical_to_encode_ = self.low_cardinality_features_
        else:
            raise ValueError(f"Unknown high_cardinality_strategy: {self.high_cardinality_strategy}")

        if self.verbose:
            print(f"Detected numerics: {len(self.initial_numerical_features_)}; categoricals: {len(self.initial_categorical_features_)}")
            print(f"Low-cardinality => OHE/label: {self.low_cardinality_features_}")
            print(f"High-cardinality => strategy {self.high_cardinality_strategy}: {self.high_cardinality_features_}")
            print("--- AutoPreprocessor fit complete ---")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.verbose: print("--- Starting AutoPreprocessor transform ---")
        df = X.copy()
        # drop id/date features
        to_drop = self.date_features_ + self.id_features_
        df.drop(columns=[c for c in to_drop if c in df.columns], inplace=True)
        # impute & create indicators
        for col, val in self.imputation_values_.items():
            if col in df.columns:
                missing_mask = df[col].isnull()
                if missing_mask.any():
                    df[col].fillna(val, inplace=True)
                indicator_col = f"{col}_was_missing"
                df[indicator_col] = missing_mask.astype(int)

        parts = []
        numeric_cols = [c for c in self.initial_numerical_features_ if c in df.columns]
        missing_indicators = [c for c in df.columns if c.endswith('_was_missing')]
        parts.append(df[numeric_cols + missing_indicators].reset_index(drop=True))

        # low-cardinality: one-hot
        if self.low_cardinality_features_:
            low = [c for c in self.low_cardinality_features_ if c in df.columns]
            if low:
                df_low = df[low].fillna("missing_category")
                df_low_ohe = pd.get_dummies(df_low, columns=low, drop_first=True)
                parts.append(df_low_ohe.reset_index(drop=True))

        # high-cardinality
        if self.high_cardinality_features_:
            high = [c for c in self.high_cardinality_features_ if c in df.columns]
            if high:
                df_high = df[high].copy()
                if self.high_cardinality_strategy == 'grouping':
                    for col in high:
                        frequent = self.frequent_categories_.get(col, [])
                        df_high[col] = df_high[col].where(df_high[col].isin(frequent), 'rare_category')
                    df_high_ohe = pd.get_dummies(df_high, columns=high, drop_first=True)
                    parts.append(df_high_ohe.reset_index(drop=True))
                elif self.high_cardinality_strategy == 'hashing' and self.hasher_:
                    dict_rows = df_high.fillna("").to_dict(orient='records')
                    hashed = self.hasher_.transform(dict_rows).toarray()
                    df_hashed = pd.DataFrame(hashed, columns=[f"hash_{i}" for i in range(hashed.shape[1])])
                    parts.append(df_hashed.reset_index(drop=True))

        X_transformed = pd.concat(parts, axis=1)
        if not self.final_features_:
            self.final_features_ = X_transformed.columns.tolist()
        X_transformed = X_transformed.reindex(columns=self.final_features_, fill_value=0)

        if self.verbose: print(f"--- Transform complete. Final shape: {X_transformed.shape} ---")
        return X_transformed

    def _detect_feature_types(self, df: pd.DataFrame):
        # numeric-like strings
        for col in self.initial_categorical_features_[:]:
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                converted = pd.to_numeric(df[col], errors='coerce')
                if converted.notnull().sum() == df[col].notnull().sum():
                    self.initial_categorical_features_.remove(col)
                    self.initial_numerical_features_.append(col)

        # date-like
        for col in self.initial_categorical_features_[:]:
            if df[col].isnull().all(): continue
            converted = pd.to_datetime(df[col], errors='coerce')
            if converted.notna().sum() / df[col].notna().sum() > 0.95:
                self.date_features_.append(col)
                self.initial_categorical_features_.remove(col)

        # categorical IDs
        for col in self.initial_categorical_features_[:]:
            if df[col].nunique(dropna=False) / len(df) > self.id_threshold:
                self.id_features_.append(col)
                self.initial_categorical_features_.remove(col)

        # numeric IDs
        for col in self.initial_numerical_features_[:]:
            if df[col].nunique(dropna=False) / len(df) > self.id_threshold:
                is_float_like = (df[col].dropna() % 1 != 0).any()
                if not is_float_like:
                    self.id_features_.append(col)
                    self.initial_numerical_features_.remove(col)


class ModelAwarePreprocessor(AutoPreprocessor):
    """
    Model-aware wrapper over AutoPreprocessor.

    configure_from_profiler(report) will set encoding / scaling strategy:
    - For linear (Parametric): prefer grouping for high-card and One-Hot for low-card,
      + StandardScaler for numerical features.
    - For non-linear (Non-Parametric): prefer hashing for high-card, label-encoding for low-card,
      no scaling. Also provide categorical_feature_names_ for CatBoost if desired.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler_: Optional[StandardScaler] = None
        self.scale_numeric: bool = False
        self.low_card_label_encode_: List[str] = []
        self.categorical_feature_names_: List[str] = []  # for CatBoost-like use

    def configure_from_profiler(self, profiler_report: Dict[str, Any]):
        """
        Configure preprocessing behavior using profiler report (output of ModelDataProfiler.profile()).
        Expects profiler_report to be the 'report' dict returned by ModelDataProfiler.profile().
        """
        diag = profiler_report.get("diagnostics", {})
        rec_type = diag.get("recommendation", None) or profiler_report.get("diagnostics", {}).get("recommendation_type", None)

        # Normalize possible values
        # If profiler used earlier convention 'Parametric'/'Non-Parametric' in 'recommendation_type',
        # or used 'recommendation' string, try to interpret.
        if isinstance(rec_type, str):
            rt = rec_type.lower()
        elif rec_type is None:
            rt = None
        else:
            rt = str(rec_type).lower()

        # Default to non-linear if not clear
        if "parametric" in rt or "linear" in rt:
            # linear workflow
            if self.verbose: print("ModelAwarePreprocessor: configuring for LINEAR models (parametric).")
            self.high_cardinality_strategy = 'grouping'
            self.rare_category_threshold = getattr(self, 'rare_category_threshold', 0.01)
            self.scale_numeric = True
            self.scaler_ = StandardScaler()
            # keep low-cardinality as OHE (AutoPreprocessor default)
            self.low_card_label_encode_ = []
            self.categorical_feature_names_ = []
        else:
            # non-linear / tree-based workflow
            if self.verbose: print("ModelAwarePreprocessor: configuring for TREE-BASED models (non-parametric).")
            self.high_cardinality_strategy = 'hashing'
            # force rare threshold to 0 to ensure hashing of many categories if desired
            self.rare_category_threshold = 0.0
            self.n_hashing_features = max(16, getattr(self, 'n_hashing_features', 20))
            self.scale_numeric = False
            # For low-cardinality prefer label encoding (trees like it)
            self.low_card_label_encode_ = self.low_cardinality_features_.copy()
            # CatBoost prefers native categorical names (we'll supply list)
            self.categorical_feature_names_ = self.low_cardinality_features_ + self.high_cardinality_features_

# --- within ModelAwarePreprocessor class ---

    def fit(self, X: pd.DataFrame, y=None):
        # run AutoPreprocessor fit first
        super().fit(X, y)
        # if scaling desired, fit scaler on numeric columns
        if self.scale_numeric:
            numeric_cols = [c for c in self.initial_numerical_features_ if c not in self.id_features_]
            if numeric_cols:
                self.scaler_ = StandardScaler()
                # store the exact column order used for scaler fitting so we can reproduce it
                self.scaler_feature_names_ = list(numeric_cols)
                # Fit the scaler on the DataFrame values in that exact order
                scaler_input = X.reindex(columns=self.scaler_feature_names_, fill_value=0).fillna(0).values
                self.scaler_.fit(scaler_input)
        return self

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[str]]]:
        df = X.copy()
        # create imputations and indicators, etc. (parent transform handles basic encodings for OHE/hashing)
        pre_trans = super().transform(df)

        # If we requested label-encoding for low-cardinality features (tree workflow),
        # apply label encoding in place of OHE part...
        # (unchanged from previous code) ...
        if self.low_card_label_encode_:
            # ... same label-encoding assembly ...
            numeric_and_missing = pre_trans[self.initial_numerical_features_ + [c for c in pre_trans.columns if c.endswith('_was_missing') and c not in self.initial_numerical_features_]].copy()
            df_low = X[self.low_card_label_encode_].fillna("missing_category").astype(str).copy()
            le_parts = []
            for c in df_low.columns:
                le = LabelEncoder()
                try:
                    le.fit(df_low[c])
                    encoded = le.transform(df_low[c])
                except Exception:
                    encoded = pd.factorize(df_low[c])[0]
                le_parts.append(pd.Series(encoded, name=c))
            le_df = pd.concat(le_parts, axis=1)
            hashed_cols = [c for c in pre_trans.columns if c.startswith("hash_")]
            hashed_df = pre_trans[hashed_cols].copy() if hashed_cols else pd.DataFrame(index=pre_trans.index)
            transformed = pd.concat([numeric_and_missing.reset_index(drop=True), le_df.reset_index(drop=True), hashed_df.reset_index(drop=True)], axis=1)
        else:
            transformed = pre_trans

        # apply scaling if configured
        if self.scale_numeric and getattr(self, "scaler_", None) is not None and getattr(self, "scaler_feature_names_", None):
            # Ensure we use exactly the same column names and order as during fit
            numeric_cols = [c for c in self.scaler_feature_names_ if c in transformed.columns]
            # Build array in the exact order (missing columns filled with zeros)
            scaler_input_df = transformed.reindex(columns=self.scaler_feature_names_, fill_value=0).fillna(0)
            scaler_input = scaler_input_df.values
            scaled_array = self.scaler_.transform(scaler_input)
            # Replace the numeric columns in the transformed DataFrame with scaled values
            transformed.loc[:, self.scaler_feature_names_] = scaled_array

        # If CatBoost-style output requested, return categorical feature names for native use
        if self.categorical_feature_names_:
            return transformed, self.categorical_feature_names_
        return transformed