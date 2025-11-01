# src/preprocessing.py
import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Dict, Any, Union, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import sparse as sp

class AutoPreprocessor(BaseEstimator, TransformerMixin):
    """
    A preprocessor that automates feature detection, cleaning, imputation,
    and encoding. Supports returning a sparse transform (hashed part as sparse)
    to save memory for very high-cardinality features.
    """
    def __init__(self, id_threshold: float = 0.95,
                 cardinality_threshold: int = 50,
                 high_cardinality_strategy: str = 'grouping',
                 n_hashing_features: int = 20,
                 rare_category_threshold: float = 0.01,
                 output_sparse: bool = True,
                 verbose: bool = True):
        self.id_threshold = id_threshold
        self.cardinality_threshold = cardinality_threshold
        self.high_cardinality_strategy = high_cardinality_strategy
        self.n_hashing_features = n_hashing_features
        self.rare_category_threshold = rare_category_threshold
        self.output_sparse = output_sparse
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
        self.low_cardinality_features_ = []
        self.high_cardinality_features_ = []
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
            # for hashing, we will encode low-cardinals separately
            self.categorical_to_encode_ = self.low_cardinality_features_
        else:
            raise ValueError(f"Unknown high_cardinality_strategy: {self.high_cardinality_strategy}")

        if self.verbose:
            print(f"Detected numerics: {len(self.initial_numerical_features_)}; categoricals: {len(self.initial_categorical_features_)}")
            print(f"Low-cardinality => OHE/label: {self.low_cardinality_features_}")
            print(f"High-cardinality => strategy {self.high_cardinality_strategy}: {self.high_cardinality_features_}")
            print("--- AutoPreprocessor fit complete ---")
        return self

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, Tuple[Union[pd.DataFrame, sp.spmatrix], List[str]]]:
        """
        Returns:
         - If output_sparse True: (csr_matrix, feature_names) where feature_names corresponds to columns in the matrix.
         - If output_sparse False: pandas DataFrame (dense).
        The transform will always respect self.final_features_ if it's already been set during a prior transform,
        ensuring transforms on different splits produce matrices with identical column ordering and width.
        """
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
                    # convert categorical to object before fill to avoid Categorical assignment errors
                    if pd.api.types.is_categorical_dtype(df[col]):
                        df[col] = df[col].astype(object)
                    df[col].fillna(val, inplace=True)
                indicator_col = f"{col}_was_missing"
                df[indicator_col] = missing_mask.astype(int)

        parts_dense = []
        dense_feature_names: List[str] = []

        # Part A: numeric + missing indicators (dense)
        numeric_cols = [c for c in self.initial_numerical_features_ if c in df.columns]
        missing_indicators = [c for c in df.columns if c.endswith('_was_missing')]
        dense_part_cols = numeric_cols + missing_indicators
        if dense_part_cols:
            parts_dense.append(df[dense_part_cols].reset_index(drop=True))
            dense_feature_names.extend(dense_part_cols)

        # Part B: low-cardinality one-hot (dense)
        if self.low_cardinality_features_:
            low = [c for c in self.low_cardinality_features_ if c in df.columns]
            if low:
                # ensure categorical columns converted to object before fillna to avoid categorical setitem errors
                df_low = df[low].copy()
                for c in df_low.columns:
                    if pd.api.types.is_categorical_dtype(df_low[c]):
                        df_low[c] = df_low[c].astype(object)
                df_low = df_low.fillna("missing_category")
                df_low_ohe = pd.get_dummies(df_low, columns=low, drop_first=True)
                parts_dense.append(df_low_ohe.reset_index(drop=True))
                dense_feature_names.extend(df_low_ohe.columns.tolist())

        # Build dense block (if any)
        if parts_dense:
            dense_df = pd.concat(parts_dense, axis=1)
            # ensure deterministic column ordering
            dense_df.columns = _make_unique(list(dense_df.columns))
            # downcast where possible to float32
            try:
                dense_df = dense_df.astype(np.float32, errors='ignore')
            except Exception:
                pass
        else:
            dense_df = pd.DataFrame(index=df.index)

        # Part C: high-cardinality handling (hashed -> sparse) or grouped -> dense
        sparse_hashed = None
        hashed_feature_names = []
        if self.high_cardinality_features_:
            high = [c for c in self.high_cardinality_features_ if c in df.columns]
            if high:
                if self.high_cardinality_strategy == 'grouping':
                    df_high = df[high].copy()
                    for col in high:
                        frequent = self.frequent_categories_.get(col, [])
                        df_high[col] = df_high[col].where(df_high[col].isin(frequent), 'rare_category')
                    df_high_ohe = pd.get_dummies(df_high, columns=high, drop_first=True).reset_index(drop=True)
                    df_high_ohe.columns = _make_unique(list(df_high_ohe.columns))
                    if not dense_df.empty:
                        dense_df = pd.concat([dense_df.reset_index(drop=True), df_high_ohe.reset_index(drop=True)], axis=1)
                    else:
                        dense_df = df_high_ohe
                    dense_df = dense_df.astype(np.float32, errors='ignore')
                    dense_feature_names = list(dense_df.columns)
                elif self.high_cardinality_strategy == 'hashing':
                    df_high = df[high].copy().fillna("").astype(str)
                    dict_rows = df_high.to_dict(orient='records')
                    if self.hasher_ is None:
                        self.hasher_ = FeatureHasher(n_features=self.n_hashing_features, input_type='dict')
                    hashed_sparse = self.hasher_.transform(dict_rows)  # scipy.sparse matrix
                    sparse_hashed = sp.csr_matrix(hashed_sparse, dtype=np.float32)
                    hashed_feature_names = [f"hash_{i}" for i in range(sparse_hashed.shape[1])]

        # If output_sparse requested, convert dense to sparse and hstack --- ensure final_features_ stability
        if self.output_sparse:
            # Determine final feature ordering: use cached self.final_features_ if present; otherwise create and persist it
            if not self.final_features_:
                # finalize feature names: dense then hashed
                self.final_features_ = list(dense_feature_names) + hashed_feature_names

            final_dense_names = [n for n in self.final_features_ if not n.startswith("hash_")]
            final_hashed_names = [n for n in self.final_features_ if n.startswith("hash_")]
            # Build dense_sparse in the order of final_dense_names
            if final_dense_names:
                # ensure dense_df has these columns (fill missing with zeros)
                dense_aligned = dense_df.reindex(columns=final_dense_names, fill_value=0)
                dense_sparse = sp.csr_matrix(dense_aligned.fillna(0).values.astype(np.float32))
            else:
                dense_sparse = None

            # Build hashed sparse aligned to final_hashed_names count
            if final_hashed_names:
                expected_hash_cols = len(final_hashed_names)
                if sparse_hashed is None:
                    # no hashed part present in this call but expected: create zero sparse
                    sparse_hashed_aligned = sp.csr_matrix((len(df), expected_hash_cols), dtype=np.float32)
                else:
                    # if existing hashed matrix has different number of cols, pad or trim as needed
                    if sparse_hashed.shape[1] == expected_hash_cols:
                        sparse_hashed_aligned = sparse_hashed
                    elif sparse_hashed.shape[1] < expected_hash_cols:
                        # pad with zero columns
                        pad = sp.csr_matrix((sparse_hashed.shape[0], expected_hash_cols - sparse_hashed.shape[1]), dtype=np.float32)
                        sparse_hashed_aligned = sp.hstack([sparse_hashed, pad], format='csr')
                    else:
                        # trim extra columns
                        sparse_hashed_aligned = sparse_hashed[:, :expected_hash_cols]
            else:
                sparse_hashed_aligned = None

            # hstack dense + hashed
            if dense_sparse is not None and sparse_hashed_aligned is not None:
                X_sparse = sp.hstack([dense_sparse, sparse_hashed_aligned], format='csr')
            elif sparse_hashed_aligned is not None:
                X_sparse = sparse_hashed_aligned
            elif dense_sparse is not None:
                X_sparse = dense_sparse
            else:
                X_sparse = sp.csr_matrix((len(df), 0), dtype=np.float32)

            if self.verbose:
                print(f"--- Transform complete. Sparse shape: {X_sparse.shape} (features={len(self.final_features_)}) ---")
            return X_sparse, self.final_features_

        # else return dense DataFrame (merge dense and hashed if grouping produced dense)
        if not dense_df.empty:
            X_out = dense_df.reset_index(drop=True)
        else:
            X_out = pd.DataFrame(index=df.index)
        if sparse_hashed is not None:
            hashed_dense = pd.DataFrame(sparse_hashed.toarray(), columns=hashed_feature_names).reset_index(drop=True)
            X_out = pd.concat([X_out.reset_index(drop=True), hashed_dense.reset_index(drop=True)], axis=1)
        X_out.columns = _make_unique(list(X_out.columns))
        if not self.final_features_:
            self.final_features_ = X_out.columns.tolist()
        X_out = X_out.reindex(columns=self.final_features_, fill_value=0)
        numeric_cols = X_out.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            X_out[numeric_cols] = X_out[numeric_cols].astype(np.float32)
        if self.verbose:
            print(f"--- Transform complete. Final shape: {X_out.shape} ---")
        return X_out

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


def _make_unique(cols: List[str]) -> List[str]:
    """Make list of column names unique by appending suffixes to duplicates (preserving order)."""
    seen = {}
    out = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            new_name = f"{c}__dup{seen[c]}"
            while new_name in seen:
                seen[c] += 1
                new_name = f"{c}__dup{seen[c]}"
            seen[new_name] = 0
            out.append(new_name)
    return out

def _make_unique(cols: List[str]) -> List[str]:
    """Make list of column names unique by appending suffixes to duplicates (preserving order)."""
    seen = {}
    out = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            new_name = f"{c}__dup{seen[c]}"
            # ensure new_name is also unique
            while new_name in seen:
                seen[c] += 1
                new_name = f"{c}__dup{seen[c]}"
            seen[new_name] = 0
            out.append(new_name)
    return out


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
        self._profiler_preference_: Optional[str] = None
        self.scaler_feature_names_: List[str] = []

    def configure_from_profiler(self, profiler_report: Dict[str, Any]):
        """
        Configure preprocessing preference using profiler report.
        We DO NOT apply strategy changes that depend on lists (low/high cardinality) here,
        because those lists are populated during fit. Instead we record the preference and
        apply it in fit().
        """
        diag = profiler_report.get("diagnostics", {}) or {}
        # prefer explicit keys in diagnostics
        rec = diag.get("recommendation_type") or diag.get("recommendation") or ""
        rt = str(rec).lower() if rec is not None else ""

        # detect 'non-param' first (unambiguous)
        if "non-param" in rt or "non param" in rt or ("non" in rt and "param" in rt):
            pref = "non-linear"
        elif "param" in rt or "linear" in rt:
            pref = "linear"
        else:
            # fallback: check top-level profiler model_type field
            top_model_type = profiler_report.get("model_type", "") or ""
            pref = "linear" if "param" in str(top_model_type).lower() or "linear" in str(top_model_type).lower() else "non-linear"

        self._profiler_preference_ = pref
        if self.verbose:
            print(f"ModelAwarePreprocessor: recorded profiler preference -> {self._profiler_preference_} (will apply during fit)")

    def fit(self, X: pd.DataFrame, y=None):
        # run AutoPreprocessor fit first (this populates low/high card lists)
        super().fit(X, y)

        # Apply profiler preference now that we have low/high cardinal lists
        pref = getattr(self, "_profiler_preference_", None)
        if pref == "linear":
            if self.verbose: print("Applying LINEAR (parametric) preprocessing configuration.")
            self.high_cardinality_strategy = 'grouping'
            # keep rare_category_threshold as-is (or default)
            self.scale_numeric = True
            self.low_card_label_encode_ = []  # prefer OHE for low-card
            self.categorical_feature_names_ = []
        else:
            if self.verbose: print("Applying NON-LINEAR (tree) preprocessing configuration.")
            self.high_cardinality_strategy = 'hashing'
            self.rare_category_threshold = 0.0
            self.n_hashing_features = max(16, getattr(self, 'n_hashing_features', 20))
            self.scale_numeric = False
            # now that low_cardinality_features_ is set by super().fit, set label encode targets
            self.low_card_label_encode_ = self.low_cardinality_features_.copy()
            self.categorical_feature_names_ = self.low_cardinality_features_ + self.high_cardinality_features_

        # If scaling desired, fit scaler on numeric columns and persist column order (deduplicated)
        if self.scale_numeric:
            numeric_cols = [c for c in self.initial_numerical_features_ if c not in self.id_features_]
            # Deduplicate numeric_cols preserving order
            seen = set()
            numeric_cols_unique = []
            for c in numeric_cols:
                if c not in seen:
                    seen.add(c)
                    numeric_cols_unique.append(c)
            if numeric_cols_unique:
                self.scaler_ = StandardScaler()
                self.scaler_feature_names_ = numeric_cols_unique
                # Fit the scaler on the DataFrame values in that exact order
                scaler_input = X.reindex(columns=self.scaler_feature_names_, fill_value=0).fillna(0).values
                self.scaler_.fit(scaler_input)
        else:
            self.scaler_ = None
            self.scaler_feature_names_ = []

        return self

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[str]]]:
        df = X.copy()
        # create imputations and indicators, etc. (parent transform handles basic encodings for OHE/hashing)
        pre_trans = super().transform(df)

        if isinstance(pre_trans, tuple):
            # pre_trans expected forms:
            # - (X_sparse, feature_names)  where X_sparse is scipy.sparse CSR and feature_names is list[str]
            # - or (X_sparse, dense_feature_names, hashed_feature_names) if you extended AutoPreprocessor
            # For now, keep behavior simple:
            #  - If user requested low-cardinality label encoding for tree workflows, we *prefer* label-encoding,
            #    however constructing label-encoded + sparse hashed assembly may be memory-heavy. We therefore
            #    fallback to the sparse output produced by AutoPreprocessor (which used OHE for low-card features).
            if self.low_card_label_encode_:
                if self.verbose:
                    print("⚠️ low-cardinality label-encoding requested but preprocessor returned sparse output. "
                          "Falling back to sparse/OHE output (skipping label-encoding to avoid excessive memory use).")
            # Return the sparse tuple unchanged (caller code expects a tuple when output_sparse is True)
            return pre_trans
        
        # If we requested label-encoding for low-cardinality features (tree workflow),
        # apply label encoding in place of OHE part. Because AutoPreprocessor by default
        # will have appended OHE for low-card; we need to re-create label-encoded low-card
        # representation if configured that way. A simpler approach: rebuild label-encoded
        # representation from original df for those low-card features and then stitch with numeric part.
        if getattr(self, "low_card_label_encode_", None):
            # numeric_and_missing: collect numeric and missing-indicators from pre_trans
            numeric_and_missing = []
            for c in self.initial_numerical_features_:
                if c in pre_trans.columns:
                    numeric_and_missing.append(c)
            numeric_and_missing += [c for c in pre_trans.columns if c.endswith('_was_missing') and c not in numeric_and_missing]
            numeric_and_missing_df = pre_trans[numeric_and_missing].copy() if numeric_and_missing else pd.DataFrame(index=pre_trans.index)

            # label encode low-card features from original df (fit on missing->'missing_category')
            df_low = X[self.low_card_label_encode_].fillna("missing_category").astype(str).copy() if self.low_card_label_encode_ else pd.DataFrame(index=pre_trans.index)
            le_parts = []
            for c in df_low.columns:
                le = LabelEncoder()
                try:
                    le.fit(df_low[c])
                    encoded = le.transform(df_low[c])
                except Exception:
                    encoded = pd.factorize(df_low[c])[0]
                le_parts.append(pd.Series(encoded, name=c))
            le_df = pd.concat(le_parts, axis=1) if le_parts else pd.DataFrame(index=pre_trans.index)

            # For high-cardinality hashed features, keep the hashed columns appended by AutoPreprocessor (if any)
            hashed_cols = [c for c in pre_trans.columns if c.startswith("hash_")]
            hashed_df = pre_trans[hashed_cols].copy() if hashed_cols else pd.DataFrame(index=pre_trans.index)

            transformed = pd.concat([numeric_and_missing_df.reset_index(drop=True), le_df.reset_index(drop=True), hashed_df.reset_index(drop=True)], axis=1)
        else:
            transformed = pre_trans

        # Ensure column names are unique to avoid reindex error (when transform produced duplicates)
        transformed.columns = _make_unique(list(transformed.columns))

        # apply scaling if configured and scaler was fitted
        if self.scale_numeric and getattr(self, "scaler_", None) is not None and getattr(self, "scaler_feature_names_", None):
            # Ensure we use exactly the same column names and order as during fit
            # If transformed has duplicates or missing numeric columns, reindexing will be safe because we made names unique
            scaler_cols = [c for c in self.scaler_feature_names_]
            # Build array in the exact order (missing columns filled with zeros)
            scaler_input_df = transformed.reindex(columns=scaler_cols, fill_value=0).fillna(0)
            # If duplicates remain for some reason, take the first occurrence per column name
            if scaler_input_df.columns.duplicated().any():
                scaler_input_df = scaler_input_df.loc[:, ~scaler_input_df.columns.duplicated()]
            scaler_input = scaler_input_df.values
            try:
                scaled_array = self.scaler_.transform(scaler_input)
                # Replace the numeric columns in the transformed DataFrame with scaled values
                transformed.loc[:, scaler_cols] = scaled_array
            except Exception as e:
                # fallback: skip scaling and warn
                if self.verbose:
                    print(f"⚠️ Scaling failed during transform: {e}. Proceeding without scaling.")

        # If CatBoost-style output requested, return categorical feature names for native use
        if self.categorical_feature_names_:
            return transformed, self.categorical_feature_names_
        return transformed