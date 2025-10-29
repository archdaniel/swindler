# will identify, based on data, which is the most defining bit here.
# lets study the possibility of using the evidently pack. 
# only making pairwise comparisons ignores if a linear combination of features can predict a target. 

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan, normal_ad
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, log_loss
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.stattools import durbin_watson
from statsmodels.api import OLS, Logit, add_constant
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
import warnings
from pathlib import Path
from typing import Union, Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import pandas as pd
from nni.algorithms.feature_engineering.gradient_selector import FeatureGradientSelector, fginitialize
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import shapiro, spearmanr, levene
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import FeatureHasher
import scipy.sparse.linalg
import scipy.sparse
import warnings
warnings.filterwarnings("ignore")
seed = 505

def safe_compute_data_stats(self):
    X, y = self.X[self.ix_statistics], self.y[self.ix_statistics]
    preprocess = self.preprocess
    classification = self.classification

    Xmn = X.mean(dim=0).numpy()
    Xsd = X.std(dim=0).numpy()
    Xc = (X - Xmn) / (Xsd + 1e-8)
    Xc_np = Xc.detach().cpu().numpy()   # ensure numpy

    sv1 = scipy.sparse.linalg.svds(Xc_np / np.sqrt(y.numel()), k=1, which='LM', return_singular_vectors=False)
    sv1 = np.array([min(np.finfo(np.float32).max, sv1[0])])

    ymn, ysd = (0., 1.) if classification else (y.mean().item(), y.std().item())
    return Xmn, sv1, Xsd, ymn, ysd

fginitialize.PrepareData.compute_data_stats = safe_compute_data_stats


class DataProfiler:
    """
    A class to profile a dataset from a pandas DataFrame, CSV, or Parquet file.

    This profiler provides insights into:
    - Correlations between numerical features (Pearson and Kendall).
    - Normality of numerical feature distributions (Shapiro-Wilk test).
    - The domain of each feature (min/max for numerical, unique values for categorical).
    - This function uses a Naive isolated check.
    - Correlation(Xᵢ, Y) or Mutual Information(Xᵢ, Y) you’re only testing individual features one by one. That means you’re checking if each Xᵢ → Y relationship is linear (or nonlinear) in isolation. But in reality, the target may depend on a combination of features.
    
    """

    def __init__(self, data_source: Union[pd.DataFrame, str, Path]):
        """
        Initializes the DataProfiler with a data source.

        Args:
            data_source (Union[pd.DataFrame, str, Path]):
                A pandas DataFrame, or a file path to a CSV or Parquet file.
        """
        self.df = self._load_data(data_source)
        self.profile: Dict[str, Any] = {}

    def _load_data(self, data_source: Union[pd.DataFrame, str, Path], verbose: bool = True) -> pd.DataFrame:
        """Loads data from various sources into a pandas DataFrame."""
        if isinstance(data_source, pd.DataFrame):
            if verbose: print("Data source is a pandas DataFrame. Creating a copy.")
            return data_source.copy()

        if verbose: print(f"Loading data from file: {data_source}")
        path = Path(data_source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix == '.csv':
            return pd.read_csv(path)
        elif path.suffix in ['.parquet', '.pq']:
            try:
                return pd.read_parquet(path)
            except ImportError:
                raise ImportError(
                    "pyarrow is required to read parquet files. "
                    "Please install it with 'pip install pyarrow'."
                )
        else:
            raise ValueError(
                "Unsupported file format. Please provide a pandas DataFrame, "
                "or a path to a .csv or .parquet file."
            )

    def _get_feature_types(self) -> tuple[List[str], List[str]]:
        """Identifies numerical and categorical columns in the DataFrame."""
        numerical_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = self.df.select_dtypes(exclude=np.number).columns.tolist()
        return numerical_cols, categorical_cols

    def _calculate_correlations(self, numerical_cols: List[str], verbose: bool = True):
        """Calculates Pearson and Kendall correlations for numerical columns."""
        if verbose: print("Calculating correlations...")
        numerical_df = self.df[numerical_cols]
        if numerical_df.empty:
            self.profile['correlations'] = {'pearson': {}, 'kendall': {}}
            if verbose: print("No numerical columns to calculate correlations.")
            return

        pearson_corr = numerical_df.corr(method='pearson')
        kendall_corr = numerical_df.corr(method='kendall')

        self.profile['correlations'] = {
            'pearson': pearson_corr.to_dict(),
            'kendall': kendall_corr.to_dict()
        }
        if verbose: print("Finished calculating correlations.")

    def _check_normality(self, numerical_cols: List[str], p_value_threshold: float = 0.05, verbose: bool = True):
        """
        Performs Shapiro-Wilk test for normality on numerical columns.
        A feature's distribution is considered 'parametric' (normal) if the p-value
        is above the threshold.
        """
        if verbose: print("Checking for normality...")
        normality_results = {}
        for col in numerical_cols:
            # The Shapiro-Wilk test requires at least 3 data points and works best on samples up to 5000.
            data = self.df[col].dropna()
            if len(data) >= 3:
                # For larger datasets, we take a sample to avoid performance issues.
                sample = data.sample(n=min(len(data), 4999), random_state=seed)
                stat, p_value = stats.shapiro(sample)
                is_normal = p_value > p_value_threshold
                normality_results[col] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'is_normal': is_normal
                }
            else:
                normality_results[col] = {
                    'statistic': None,
                    'p_value': None,
                    'is_normal': False,
                    'notes': 'Not enough data to perform normality test.'
                }
        self.profile['normality'] = normality_results
        if verbose: print("Finished checking for normality.")

    def _describe_domain(self, numerical_cols: List[str], categorical_cols: List[str], verbose: bool = True):
        """Describes the domain for each feature."""
        domain_info = {}
        if verbose: print("Describing feature domains...")
        for col in numerical_cols:
            domain_info[col] = {
                'type': 'numerical',
                'min': self.df[col].min(),
                'max': self.df[col].max()
            }
        for col in categorical_cols:
            unique_values = self.df[col].unique()
            domain_info[col] = {
                'type': 'categorical',
                'unique_values': unique_values.tolist()
            }
        self.profile['domain'] = domain_info
        if verbose: print("Finished describing feature domains.")

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Runs the full data profiling analysis.

        Returns:
            Dict[str, Any]: A dictionary containing the full data profile.
        """
        if verbose: print("Starting DataProfiler run...")
        numerical_cols, categorical_cols = self._get_feature_types()
        self._calculate_correlations(numerical_cols, verbose=verbose)
        self._check_normality(numerical_cols, verbose=verbose)
        self._describe_domain(numerical_cols, categorical_cols, verbose=verbose)
        if verbose: print("DataProfiler run finished.")
        return self.profile

class ModelDataProfiler:
    """
    A class to profile a dataset from a pandas DataFrame, CSV, or Parquet file.

    This profiler provides insights into:
   - Normality
   - Skewness & Kurtosis
   - ANOVA (categorical target)
   - Correlation (numeric target)
   - Homoscedasticity & Multicollinearity
   - Summarize diagnostics
   - Return final recommendation

    """
    def __init__(self, data, target, categorical_features_order=None, verbose=True,
                 high_cardinality_strategy: str = 'hashing',
                 cardinality_threshold: int = 50,
                 n_hashing_features: int = 50,
                 rare_category_threshold: float = 0.00):
        self.verbose = verbose
        self.data = self._load_data(data, verbose=verbose)
        self.target = target
        self.cat_order = categorical_features_order
        self.model = None
        self.results = {}
        self.high_cardinality_strategy = high_cardinality_strategy
        self.cardinality_threshold = cardinality_threshold
        self.n_hashing_features = n_hashing_features
        self.rare_category_threshold = rare_category_threshold
        # Force-hash-all flag (explicit numeric check — don't rely on truthiness of 0.0)
        self.force_hash_all = (self.rare_category_threshold <= 0.0)

        # Attributes for strategies
        self.low_cardinality_features_: List[str] = []
        self.high_cardinality_features_: List[str] = []
        self.frequent_categories_: Dict[str, List[str]] = {}
        self.hasher_: FeatureHasher = None
        self.features_to_drop_: List[str] = []

        self._identify_feature_types()

    def _identify_feature_types(self):
        """Identifies numerical and categorical features from the dataframe."""
        if self.verbose: print("Automatically identifying feature types...")
        numerical_features = self.data.select_dtypes(include=np.number).columns.tolist()
        if self.target in numerical_features:
            numerical_features.remove(self.target)
        self.numerical_features = numerical_features
        self.categorical_features = self.data.select_dtypes(exclude=np.number).columns.tolist()
        if self.verbose: print(f"Identified {len(self.numerical_features)} numerical and {len(self.categorical_features)} categorical features.")

    def _load_data(self, data, verbose: bool = True):
        if isinstance(data, pd.DataFrame):
            if verbose: print("Loading data for ModelDataProfiler from DataFrame...")
            return data.copy()
        elif isinstance(data, str):
            if verbose: print(f"Loading data for ModelDataProfiler from file: {data}...")
            if data.endswith(".csv"):
                return pd.read_csv(data)
            elif data.endswith(".parquet"):
                return pd.read_parquet(data)
            else:
                raise ValueError("Unsupported format. Use CSV or Parquet.")
        else:
            raise ValueError("Data must be DataFrame or file path.")

    def profile_data_encoding(self):
        """
        Detects, describes, and fixes encoding/storage anomalies in dataset features.
        Now preserves date-like columns as datetime dtype (not converted to numeric or one-hot encoded).
        """
        df = self.data.copy()
        issues = {}
    
        # Regex patterns
        unit_pattern = re.compile(r"^\s*([-+]?\d*\.?\d+)\s*[a-zA-Z]+")  # e.g., '36 months', '5kg'
        pct_pattern = re.compile(r"^\s*([-+]?\d*\.?\d+)\s*%$")
        curr_pattern = re.compile(r"^\s*[$€]\s*([-+]?\d*\.?\d+)")
        emp_length_pattern = re.compile(r'year|years')
        messy_numeric_pattern = re.compile(r'[<>]|\d+\s*\+\s*[a-zA-Z]+') # e.g., '< 1 year', '10+ years'
    
        for col in df.columns:
            series = df[col]
            col_issues = []
    
            # Skip target column
            if col == self.target:
                continue
    
            # --- Detect and preserve date-like columns ---
            if series.dtype == "object" or pd.api.types.is_string_dtype(series):
                converted_dates = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
                # Treat as date if at least 80% of non-null values convert successfully
                if converted_dates.notna().sum() / max(series.notna().sum(), 1) > 0.8:
                    col_issues.append("📅 Appears to be a date. Converting to datetime (kept for time-based modeling).")
                    df[col] = converted_dates
                    if self.verbose:
                        print(f"📅 Column '{col}' detected as datetime and converted.")
                    issues[col] = col_issues
                    continue  # Skip other transformations
    
            # --- Detect numeric-like strings with symbols (%, $, commas) ---
            if series.dtype == "object":
                suspicious_mask = series.astype(str).str.contains(r'[%,$€]|[0-9]+,[0-9]+', regex=True)
                if suspicious_mask.mean() > 0.1:
                    col_issues.append(
                        f"⚠️ {round(100*suspicious_mask.mean(),1)}% of entries look numeric but contain symbols (%, $, €, commas)."
                    )
    
            # --- Detect numeric + unit patterns like "36 months" ---
            if series.dtype == "object":
                num_unit_mask = series.astype(str).str.match(unit_pattern)
                if num_unit_mask.mean() > 0.1:
                    col_issues.append(
                        f"📏 {round(100*num_unit_mask.mean(),1)}% of values appear to mix numbers and units (e.g. '36 months', '5kg')."
                    )
    
            # --- Detect possible numeric stored as string ---
            if series.dtype == "object":
                cleaned_str = (
                    series.astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("%", "", regex=False)
                    .str.strip()
                )
                numeric_mask = cleaned_str.str.replace(".", "", 1).str.isnumeric()
                if numeric_mask.mean() > 0.9:
                    col_issues.append("💡 Likely numeric but stored as string (most values convertible to float).")
    
            # --- Detect symbolic encodings ---
            if series.dtype == "object":
                values = series.astype(str)
                if values.str.endswith("%").mean() > 0.5:
                    col_issues.append("🧮 Appears to store percentages as text (values ending with '%').")
                if values.str.contains(r"\$|€").mean() > 0.5:
                    col_issues.append("💰 Appears to store currency values as text (contains $ or €).")
    
            # --- Detect numeric scale mismatches ---
            if pd.api.types.is_numeric_dtype(series):
                if series.max() > 10 and series.mean() < 1:
                    col_issues.append("📊 Possible % stored as 0–100 instead of 0–1.")
                elif series.max() <= 1 and series.mean() < 0.1:
                    col_issues.append("📉 Possible % stored as 0–1 instead of 0–100.")
    
            # --- Detect mixed types ---
            if series.dtype == "object":
                unique_types = series.dropna().map(type).nunique()
                if unique_types > 1:
                    col_issues.append("⚠️ Mixed data types detected (numeric + non-numeric or inconsistent formats).")
    
            # === Fixes ===
            if series.dtype == "object":
                fixed_series = series.astype(str).str.strip()
    
                # Fix messy numeric strings like '< 1 year' or '10+ years'
                sample_matches = series.dropna().astype(str).head(20).str.contains(messy_numeric_pattern).mean()
                if sample_matches > 0.5 or series.astype(str).str.contains(emp_length_pattern).mean() > 0.5:
                    col_issues.append("🛠️ Contains complex numeric strings (e.g., '< 1', '10+ years'). Converting to numeric scale.")
                    numbers = series.astype(str).str.extract(r'(\d+\.?\d*)', expand=False).astype(float)
                    fixed_col = np.where(series.astype(str).str.contains('<', na=False), 0.5, numbers)
                    df[col] = np.where(series.astype(str).str.contains('n/a', na=False), np.nan, fixed_col)
                    issues[col] = col_issues
                    continue
    
                # Fix percentages like "7.5%" → 0.075
                fixed_series = fixed_series.apply(
                    lambda x: float(pct_pattern.match(x).group(1)) / 100
                    if isinstance(x, str) and pct_pattern.match(x)
                    else x
                )
    
                # Fix currency "$100" → 100.0
                fixed_series = fixed_series.apply(
                    lambda x: float(curr_pattern.match(x).group(1))
                    if isinstance(x, str) and curr_pattern.match(x)
                    else x
                )
    
                # Fix numeric + units "36 months" → 36.0
                fixed_series = fixed_series.apply(
                    lambda x: float(unit_pattern.match(x).group(1))
                    if isinstance(x, str) and unit_pattern.match(x)
                    else x
                )
    
                # Try coercing remaining numeric strings
                fixed_series = pd.to_numeric(fixed_series, errors='ignore')
                df[col] = fixed_series
    
            if col_issues:
                issues[col] = col_issues
    
        if self.verbose:
            print("\n=== DATA ENCODING PROFILING REPORT ===")
            for k, v in issues.items():
                print(f"\n📁 Column: {k}")
                for msg in v:
                    print(f"  - {msg}")
        self.date_features_ = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        return issues, df

    def detect_leakage_and_proxies(
        self,
        df,
        categorical_features=None,
        numerical_features=None,
        target="target",
        min_count=100,
        purity_threshold=0.95,
        accuracy_threshold=0.99,
        verbose=True
    ):
        """
        Return dict of suspicious features -> reasons/stats.
        Works for both raw categorical features and hashed numeric columns.
        """
        flags = defaultdict(list)
        y = df[target].copy()
        # If y not numeric (e.g. strings) factorize to 0/1...
        if y.dtype == 'O' or not pd.api.types.is_numeric_dtype(y):
            y_num = pd.factorize(y)[0]
        else:
            y_num = y.values
    
        # Build feature lists if not provided
        if numerical_features is None:
            numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
            numerical_features = [c for c in numerical_features if c != target]
        if categorical_features is None:
            categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
        # ---- 1) Categorical: per-category purity check ----
        for col in categorical_features:
            ser = df[col].astype("object")
            nonnull = ser.dropna()
            if len(nonnull) == 0:
                continue
            # frequency per category and positive rate
            grp = df.groupby(col)[target].agg(['count','mean']).rename(columns={'count':'n','mean':'pos_rate'})
            # consider only categories with min_count
            grp = grp[grp['n'] >= min_count]
            if grp.empty:
                continue
            max_purity = grp['pos_rate'].max()
            max_cat = grp['pos_rate'].idxmax()
            max_n = grp.loc[max_cat, 'n']
            if max_purity >= purity_threshold:
                flags[col].append({
                    'type':'category_purity',
                    'max_purity': float(max_purity),
                    'max_cat': max_cat,
                    'count_in_cat': int(max_n),
                    'message': f"Category '{col}=={max_cat}' has purity {max_purity:.3f} (n={max_n})"
                })
            # also record high mutual information
            try:
                mi = mutual_info_classif(ser.fillna("##NA##").astype(str).values.reshape(-1,1), y_num, discrete_features=True)
                if mi[0] > 0.1:  # tunable
                    flags[col].append({'type':'mutual_info','mi':float(mi[0])})
            except Exception:
                pass
    
        # ---- 2) Numerical: check deterministic mapping / point-biserial correlation ----
        for col in numerical_features:
            ser = df[col]
            nonnull = ser.dropna()
            if len(nonnull) == 0:
                continue
            # unique value -> single-class ratio
            value_counts = df.groupby(col)[target].agg(['count','mean']).rename(columns={'count':'n','mean':'pos_rate'})
            special = value_counts[value_counts['n'] >= min_count]
            if not special.empty:
                # any value with purity >= threshold?
                mv = special['pos_rate'].max()
                if mv >= purity_threshold:
                    val = special['pos_rate'].idxmax()
                    flags[col].append({
                        'type':'value_purity',
                        'value': float(val) if pd.api.types.is_numeric_dtype(val) else str(val),
                        'purity': float(mv),
                        'message': f"value {val} in {col} has purity {mv:.3f}"
                    })
            # point-biserial (numerical vs binary target)
            try:
                if len(np.unique(y_num))==2:
                    r, p = pointbiserialr(ser.fillna(ser.mean()), y_num)
                    if abs(r) > 0.9:
                        flags[col].append({'type':'point_biserial','r':float(r),'p':float(p)})
            except Exception:
                pass
    
        # ---- 3) Single-feature predictive power (fast stumps/logistic) ----
        # Check categorical via decision stump (depth=1), numeric via simple logistic
        for col in (categorical_features + numerical_features):
            Xcol = df[[col]].copy()
            # skip all-null
            if Xcol[col].dropna().empty:
                continue
            # convert categorical to codes for tree
            is_cat = col in categorical_features
            if is_cat:
                X_test = Xcol.fillna("##NA##").astype(str)
                clf = DecisionTreeClassifier(max_depth=1)  # stump
            else:
                X_test = Xcol.fillna(Xcol.mean())
                clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=200)
    
            try:
                clf.fit(X_test, y)
                acc = accuracy_score(y, clf.predict(X_test))
                if acc >= accuracy_threshold:
                    flags[col].append({'type':'single_feature_acc','acc':float(acc),
                                       'message':f"Single-feature classifier on '{col}' has accuracy {acc:.4f}"})
            except Exception:
                continue
    
        # ---- 4) Hash-bucket inspection (if user provides hashed matrix) ----
        # If hashed features are named like 'hash_0', 'hash_1', ... they will be
        # included among numerical_features. We already tested numeric predictive power above,
        # but it's useful to explicitly label them:
        for col in numerical_features:
            if isinstance(col, str) and col.startswith("hash"):
                # if flagged above, annotate reason
                if col in flags:
                    for s in flags[col]:
                        s['note'] = "hash_bucket"
        # ---- 5) Name-based heuristics ----
        suspicious_tokens = ['status','outcome','label','final','result','loan_status','decision','paid','closed']
        for col in df.columns:
            low = col.lower()
            if any(tok in low for tok in suspicious_tokens):
                flags[col].append({'type':'name_warning','message':f"Column name contains suspicious token: {col}"})
    
        # return as regular dict
        # convert lists to easier-to-inspect format
        
        return {k:list(v) for k,v in flags.items()}
        
    def _fit_baseline_model(self, X, y, verbose: bool = True):
        """Automatically select regression or classification baseline."""
        if verbose: print("Fitting baseline model...")

        if np.issubdtype(y.dtype, np.number) and y.nunique() > 10:
            model_type = "regression"
            model = LinearRegression()
            model.fit(X, y)
            preds = model.predict(X)
        else:
            model_type = "classification"
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            preds = model.predict_proba(X)[:, 1]
        if verbose: print(f"Baseline model fitted: {model_type}. X shape :{X.shape}, y NaNs: {y.isnull().sum()}")
        return model, preds, model_type
        
    def diagnose_singular_matrix(self, X):
        Xc = sm.add_constant(X)
        XtX = Xc.T @ Xc
        eigvals = np.linalg.eigvals(XtX)
        # Avoid division by zero if min eigenvalue is zero
        min_eig = np.min(eigvals) if np.min(eigvals) != 0 else np.min(eigvals) + 1e-20
        condition_number = np.max(eigvals) / min_eig
        print(f"⚙️ Condition number: {condition_number:.2e}")
        if condition_number > 1e12:
            print("⚠️ Very high condition number — near-singular matrix.")
        # Identify near-linear dependencies
        U, s, Vt = np.linalg.svd(Xc)
        near_zero = np.where(s < 1e-10)[0]
        if len(near_zero) > 0:
            print(f"Columns involved in singularities (indices): {near_zero}")
    
    def _residual_analysis(self, X, y, preds, model_type, verbose: bool = True):
        if verbose: print("Performing residual analysis...")
        results = {}
    
        if model_type == "regression":
            residuals = y - preds
            # Normality tests
            shapiro_p = stats.shapiro(residuals.sample(min(5000, len(residuals))))[1]
            ad_stat, ad_p = normal_ad(residuals)
            results['normality'] = {'shapiro_p': shapiro_p, 'anderson_darling_p': ad_p}
    
            # Homoscedasticity (Breusch-Pagan)
            X_const = sm.add_constant(preds)
            lm, lm_pvalue, f, f_pvalue = het_breuschpagan(residuals, X_const)
            results['homoscedasticity'] = {'LM pvalue': lm_pvalue, 'F pvalue': f_pvalue}
    
            # Autocorrelation (Durbin–Watson)
            dw = durbin_watson(residuals)
            results['autocorrelation'] = {'durbin_watson': dw}
    
            # Performance
            results['performance'] = {'R2': r2_score(y, preds)}
    
            # Visual plots
            self._plot_diagnostics(y, preds, residuals, model_type, verbose=verbose)
    
            # Recommendation
            if shapiro_p < 0.05 or f_pvalue < 0.05 or dw < 1.5 or dw > 2.5:
                recommendation = "⚠️ Non-parametric or robust model suggested (e.g. tree-based, Huber)"
                # recommendation = "Non-Parametric"
            else:
                recommendation = "✅ Parametric assumptions reasonable"
                # recommendation = "Parametric"
            results['recommendation'] = recommendation
    
        elif model_type == "classification":
            # For classification, check model assumptions differently
            results['performance'] = {'log_loss': log_loss(y, preds)}
        
            # --- 1. Check multicollinearity (VIF) ---
            if self.verbose:
                print("🧮 Checking multicollinearity before Logit fit... X shape:", X.shape)
        
            X_numeric = X.select_dtypes(include=[np.number]).copy()
            vif_data = pd.DataFrame()
            vif_data["feature"] = X_numeric.columns
            vif_data["VIF"] = [variance_inflation_factor(X_numeric.values, i) for i in range(len(X_numeric.columns))]
            results['multicollinearity'] = {'vif': vif_data.to_dict('records')}
        
            # --- Detect perfect correlations ---
            corr_matrix = X_numeric.corr().abs()
            perfect_pairs = []
            for col1 in corr_matrix.columns:
                for col2 in corr_matrix.columns:
                    if col1 != col2:
                        try:
                            val = float(corr_matrix.loc[col1, col2])
                            if val >= 0.9999:
                                perfect_pairs.append((col1, col2))
                        except Exception:
                            continue  # in case of malformed correlations (e.g., NaN, Series)

        
            if len(perfect_pairs) > 0:
                print(f"⚠️ Perfectly correlated column pairs: {perfect_pairs}")
        
            # --- Identify high VIF columns ---
            high_vif = vif_data[vif_data['VIF'] > 10]
            if not high_vif.empty:
                print(f"⚠️ High VIF features (possible multicollinearity):\n{high_vif}")
        
            # --- Target correlation scoring ---
            if pd.api.types.is_numeric_dtype(y):
                y_numeric = y
            else:
                y_numeric = pd.factorize(y)[0]
        
            target_corr = {}
            for col in X_numeric.columns:
                try:
                    target_corr[col] = abs(np.corrcoef(X_numeric[col], y_numeric)[0, 1])
                except Exception:
                    target_corr[col] = 0.0
        
            # --- Drop weaker feature per correlated pair ---
            unique_pairs = []
            seen = set()
            for a, b in perfect_pairs:
                if (b, a) not in seen:
                    unique_pairs.append((a, b))
                    seen.add((a, b))
        
            perfect_to_drop = []
            for a, b in unique_pairs:
                corr_a = target_corr.get(a, 0)
                corr_b = target_corr.get(b, 0)
                drop_col = a if corr_a < corr_b else b
                perfect_to_drop.append(drop_col)
        
            # --- High VIF features ---
            vif_to_drop = high_vif.loc[high_vif['VIF'].replace(np.inf, 9999) > 10, 'feature'].tolist()
        
            # --- Combine & deduplicate ---
            cols_to_drop = list(set(perfect_to_drop + vif_to_drop))
            if cols_to_drop:
                print(f"⚠️ Dropping {len(cols_to_drop)} problematic columns before fitting Logit:")
                for c in cols_to_drop[:]:
                    print(f"   - {c} (corr with target: {target_corr.get(c, 0):.3f})")
                X_numeric = X_numeric.drop(columns=cols_to_drop, errors='ignore')
                
            # --- Drop constant/near-constant columns safely ---
            constant_cols = []
            for c in X_numeric.columns:
                try:
                    # Ensure unique name access (avoids Series ambiguity)
                    col_data = X_numeric.loc[:, c]
                    # If duplicates, aggregate first column only
                    if isinstance(col_data, pd.DataFrame):
                        col_data = col_data.iloc[:, 0]
                    if col_data.nunique(dropna=False) <= 1:
                        constant_cols.append(c)
                except Exception as e:
                    if verbose:
                        print(f"⚠️ Skipping column {c} in constant check due to: {e}")
                    continue
            
            if constant_cols:
                print(f"⚠️ Dropping constant columns: {constant_cols}")
                X_numeric = X_numeric.drop(columns=constant_cols, errors='ignore')
        
            if self.verbose:
                print(f"✅ X_numeric reduced to {X_numeric.shape[1]} features after correlation-based cleaning.")

            # Persist the features to drop for external use
            self.features_to_drop_ = cols_to_drop + constant_cols

            # --- 2. Independence of errors (Durbin-Watson) ---
            try:
                self.diagnose_singular_matrix(X_numeric)
                print(f'normal print shape of X: {X_numeric.columns}')
                logit_model = sm.Logit(y, sm.add_constant(X_numeric)).fit(disp=0)
            except np.linalg.LinAlgError:
                print("⚠️ Singular matrix detected. Retrying after dropping duplicates.")
                X_numeric = X_numeric.loc[:, ~X_numeric.T.duplicated()]
                print(f'except print shape of X: {X_numeric.columns}')
                self.diagnose_singular_matrix(X_numeric)

                logit_model = sm.Logit(y, sm.add_constant(X_numeric)).fit(disp=0)
            resid_deviance_approx = np.sign(logit_model.resid_pearson) * np.sqrt(np.abs(logit_model.resid_pearson))
            pearson_residuals = logit_model.resid_pearson
            dw = durbin_watson(pearson_residuals)
            results['autocorrelation'] = {'durbin_watson': dw}

            # Recommendation
            if not high_vif.empty or dw < 1.5 or dw > 2.5:
                recommendation = "⚠️ Potential multicollinearity or error dependency. Non-linear model (e.g., tree-based) might be more robust."
                recommendation_type = "Non-Parametric"
            else:
                recommendation = "✅ Linear model assumptions seem reasonable. Logistic Regression is a good starting point."
                recommendation_type = "Parametric"
            results['recommendation'] = recommendation
            results['recommendation_type'] = recommendation_type

            # Visual plot for classification
            if verbose: print("Generating residuals vs. fitted plot for classification...")
            plt.scatter(preds, pearson_residuals, alpha=0.5)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel('Fitted probabilities')
            plt.ylabel('Pearson residuals')
            plt.title('Residuals vs Fitted (Logistic)')
            plt.show()

            # Histogram of Deviance Residuals
            if verbose: print("Generating histogram of deviance residuals...")
            plt.hist(resid_deviance_approx, bins=30, edgecolor='k', alpha=0.7)
            plt.title('Approximate Deviance Residuals (from Pearson)')
            plt.xlabel('Residuals')
            plt.ylabel('Frequency')
            plt.show()


        if verbose: print("Finished residual analysis.")
        
        return results


    def _plot_diagnostics(self, y, preds, residuals, model_type, verbose: bool = True):
        if verbose: print("Generating diagnostic plots...")
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("Residual Diagnostics", fontsize=14, weight="bold")

        # 1. Residuals vs Fitted
        axes[0, 0].scatter(preds, residuals, alpha=0.5)
        axes[0, 0].axhline(0, color="red", linestyle="--")
        axes[0, 0].set_title("Residuals vs Fitted")
        axes[0, 0].set_xlabel("Fitted Values")
        axes[0, 0].set_ylabel("Residuals")

        # 2. QQ Plot
        qqplot(residuals, line="s", ax=axes[0, 1])
        axes[0, 1].set_title("QQ Plot")

        # 3. Residual Histogram
        sns.histplot(residuals, bins=30, kde=True, ax=axes[1, 0])
        axes[1, 0].set_title("Residual Distribution")

        # 4. Leverage Plot (using influence)
        X = sm.add_constant(preds)
        model = OLS(residuals, X).fit()
        infl = model.get_influence()
        leverage = infl.hat_matrix_diag
        axes[1, 1].scatter(leverage, residuals, alpha=0.5)
        axes[1, 1].set_title("Leverage vs Residuals")
        axes[1, 1].set_xlabel("Leverage")
        axes[1, 1].set_ylabel("Residuals")

        plt.tight_layout()
        plt.show()
        if verbose: print("Finished generating diagnostic plots.")
            
    def _detect_date_features(self, df):
        """Detect columns that are likely date/time features."""
        self.date_features_ = []
        if self.verbose:
            print("📅 Detecting date-like categorical features...")
    
        for col in df.select_dtypes(exclude=np.number).columns:
            try:
                converted = pd.to_datetime(df[col], errors="coerce")
                ratio = converted.notna().sum() / df[col].notna().sum()
                if ratio > 0.95:
                    self.date_features_.append(col)
                    if self.verbose:
                        print(f"   → '{col}' detected as date feature ({ratio:.2%} convertible).")
            except Exception:
                continue
    
        return self.date_features_

    def _detect_high_cardinality_and_ids(self, df, high_cardinality_threshold=0.95):
        """Identify high-cardinality categorical keys and numeric identifiers."""
        self.unique_categorical_keys_ = []
        self.unique_numerical_ids_ = []
    
        # --- Categorical ---
        if self.verbose:
            print("🔑 Detecting high-cardinality categorical features...")
    
        for col in df.select_dtypes(exclude=np.number).columns:
            if df[col].nunique() / len(df[col]) > high_cardinality_threshold:
                self.unique_categorical_keys_.append(col)
                if self.verbose:
                    print(f"   → '{col}' marked as potential unique key.")
    
        # --- Numerical ---
        if self.verbose:
            print("🔢 Detecting potential numeric IDs...")
    
        for col in df.select_dtypes(include=np.number).columns:
            if df[col].nunique() / len(df[col]) > high_cardinality_threshold:
                skewness = df[col].skew()
                is_low_skew = abs(skewness) < 2.0
                is_float_like = (df[col].dropna() % 1 != 0).mean() > 0.01
                if is_low_skew and not is_float_like:
                    self.unique_numerical_ids_.append(col)
                    if self.verbose:
                        print(f"   → '{col}' likely an integer ID (skew={skewness:.2f}).")
    
        return self.unique_categorical_keys_, self.unique_numerical_ids_

    def _impute_missing_and_flag(self, df):
        """Impute missing values and add binary flags for imputed columns."""
        if self.verbose:
            print("🧩 Imputing missing values and adding indicators...")
    
        self.imputation_values_ = {}
        numerical_to_impute = [
            c for c in df.select_dtypes(include=np.number).columns
            if c not in self.unique_numerical_ids_
        ]
        categorical_to_impute = [
            c for c in df.select_dtypes(exclude=np.number).columns
            if c not in self.unique_categorical_keys_ and c not in self.date_features_
        ]
    
        for col in numerical_to_impute:
            if df[col].isnull().any():
                self.imputation_values_[col] = df[col].median()
    
        for col in categorical_to_impute:
            if df[col].isnull().any():
                self.imputation_values_[col] = df[col].mode(dropna=True)[0]
    
        for col, val in self.imputation_values_.items():
            if df[col].isnull().any():
                df[f"{col}_was_missing"] = df[col].isnull().astype(int)
                df[col].fillna(val, inplace=True)
                if self.verbose:
                    print(f"   → Filled '{col}' with {val}, added '{col}_was_missing'.")
    
        return df

    def _encode_categoricals(self, df):
        """Apply chosen categorical encoding strategy (grouping / hashing / one-hot)."""
        if self.verbose:
            print("🎨 Encoding categorical features...")
    
        # Reset internal lists
        self.low_cardinality_features_ = []
        self.high_cardinality_features_ = []
    
        X_parts = []
        y = df[self.target]
    
        num_feats = [
            c for c in df.select_dtypes(include=np.number).columns
            if c not in [self.target] + self.unique_numerical_ids_
        ]
        X_parts.append(df[num_feats].reset_index(drop=True))
    
        cats = [
            c for c in df.select_dtypes(exclude=np.number).columns
            if c not in self.date_features_ and c not in self.unique_categorical_keys_
        ]
    
        if not cats:
            X = pd.concat(X_parts, axis=1)
            return X, y

        # If caller requested hashing all categorical features (explicit numeric check),
        # force all categorical columns to be treated as high-cardinality.
        if self.force_hash_all:
            self.high_cardinality_features_ = cats.copy()
            self.low_cardinality_features_ = []
        else:
            for c in cats:
                if df[c].nunique() > self.cardinality_threshold:
                    self.high_cardinality_features_.append(c)
                else:
                    self.low_cardinality_features_.append(c)

        # Low-cardinality: one-hot
        if self.low_cardinality_features_:
            df_low = df[self.low_cardinality_features_].fillna("missing_category")
            df_low_ohe = pd.get_dummies(df_low, columns=self.low_cardinality_features_, drop_first=True)
            X_parts.append(df_low_ohe.reset_index(drop=True))
    
        # High-cardinality handling
        if self.high_cardinality_features_:
            if self.high_cardinality_strategy == "grouping":
                if self.verbose:
                    print("   → Grouping rare categories.")
                df_high = df[self.high_cardinality_features_].copy()
                for col in self.high_cardinality_features_:
                    counts = df_high[col].value_counts(normalize=True)
                    freq = counts[counts >= self.rare_category_threshold].index
                    df_high[col] = np.where(df_high[col].isin(freq), df_high[col], "rare_category")
                X_parts.append(pd.get_dummies(df_high, drop_first=True).reset_index(drop=True))
                
            elif self.high_cardinality_strategy == "hashing":
                if self.verbose:
                    print(f"   → Hashing {len(self.high_cardinality_features_)} features to {self.n_hashing_features} dims.")
                
                # --- Safety checks before hashing ---
                df_high = df[self.high_cardinality_features_].copy()
            
                # Convert to dict-of-records for FeatureHasher
                dict_rows = df_high.to_dict(orient="records")
        
                # Check if there is at least one non-empty value to hash (tolerant check)
                any_non_null = any(any((v is not None and (not (isinstance(v, float) and np.isnan(v)))) for v in row.values()) for row in dict_rows)
                if not any_non_null:
                    if self.verbose:
                        print("⚠️ No valid categorical values to hash — skipping hashing step.")
                else:
                    # Perform the actual hashing
                    self.hasher_ = FeatureHasher(
                        n_features=self.n_hashing_features,
                        input_type="dict"
                    )
                    try:
                        hashed = self.hasher_.fit_transform(dict_rows).toarray()
                        df_hashed = pd.DataFrame(
                            hashed,
                            columns=[f"hash_{i}" for i in range(hashed.shape[1])]
                        ).reset_index(drop=True)
                        X_parts.append(df_hashed)
                        if self.verbose:
                            print(f"✅ Successfully hashed into {df_hashed.shape[1]} features.")
                    except ValueError as e:
                        if self.verbose:
                            print(f"⚠️ Skipping hashing due to ValueError: {e}")
    
        X = pd.concat(X_parts, axis=1)
        X = X.drop(columns=[self.target] + self.date_features_, errors="ignore")
        
        return X, y

    def _detect_leakage(self, X, df):
        """Detect features with potential target leakage.

        NOTE: This method now returns the list of suspicious columns and sets self.leakage_flags.
        It does not mutate the passed `df` nor the `X` in-place; the caller should decide
        whether/when to drop the returned columns.
        """
        if self.verbose:
            print("🕵️ Checking for target leakage or proxy variables...")
    
        self.leakage_flags = self.detect_leakage_and_proxies(df, target=self.target, verbose=self.verbose)
        if not self.leakage_flags:
            return []
    
        suspicious_cols = list(self.leakage_flags.keys())
        if self.verbose:
            print(f"⚠️ Found {len(suspicious_cols)} suspected leakage features: {suspicious_cols[:5]}{'...' if len(suspicious_cols)>5 else ''}")
        # Do not drop here — return the list for the caller to apply (safer & consistent)
        return suspicious_cols

    def _fit_and_diagnose(self, X, y):
        """Fit baseline model and run residual diagnostics.

        This function will:
        - Fit an initial baseline model (regression or classification).
        - Run residual diagnostics which may identify features to drop (self.features_to_drop_).
        - If features_to_drop_ are set, drop them and refit the baseline model to return the cleaned model/predictions.
        """
        if self.verbose:
            print("🧮 Fitting baseline model and running residual diagnostics...")
    
        self.model_, preds, model_type = self._fit_baseline_model(X, y, verbose=self.verbose)
        self.diagnostics_ = self._residual_analysis(X, y, preds, model_type, verbose=self.verbose)

        # If _residual_analysis identified features to drop, apply them and refit
        if hasattr(self, "features_to_drop_") and self.features_to_drop_:
            cols_to_drop = [c for c in self.features_to_drop_ if c in X.columns]
            if cols_to_drop:
                if self.verbose:
                    print(f"⚠️ Removing {len(cols_to_drop)} features identified by residual analysis: {cols_to_drop[:5]}{'...' if len(cols_to_drop)>5 else ''}")
                X_reduced = X.drop(columns=cols_to_drop, errors="ignore")
                # Refit a final baseline model on the reduced set
                if self.verbose:
                    print("🔁 Refitting baseline model after dropping problematic columns...")
                self.model_, preds, model_type = self._fit_baseline_model(X_reduced, y, verbose=self.verbose)
                return self.model_, preds, model_type, self.diagnostics_
        
        return self.model_, preds, model_type, self.diagnostics_

    
    def profile(self):
        """Run full data and model diagnostic pipeline."""
        if self.verbose: print("🚀 Starting ModelDataProfiler run...")
    
        _, df = self.profile_data_encoding()
    
        # Sequential diagnostics
        print(f"running the _detect_date_features bit now... df shape at this moment: {df.shape}")
        self._detect_date_features(df)
        print(f"running the _detect_high_cardinality_and_ids bit now... df shape at this moment: {df.shape}")
        self._detect_high_cardinality_and_ids(df)
        print(f"running the _impute_missing_and_flag bit now... df shape at this moment: {df.shape}")
        df = self._impute_missing_and_flag(df)
        print(f"running the _encode_categoricals bit now... df shape at this moment: {df.shape}")
        # The force_hash_all flag is already computed in __init__ (explicit numeric check)
        
        X, y = self._encode_categoricals(df)
        print(f"running the _detect_leakage bit now... df shape at this moment: {df.shape}")
        suspicious_cols = self._detect_leakage(X, df)
        if suspicious_cols:
            X = X.drop(columns=[c for c in suspicious_cols if c in X.columns], errors="ignore")
            if self.verbose:
                print(f"⚠️ Removed {len(suspicious_cols)} leakage columns prior to modeling.")

        # _fit_and_diagnose will run residual analysis and — if it identifies features_to_drop_ —
        # will drop them and re-fit the final baseline model before returning.
        print(f"running the _fit_and_diagnose bit now... df shape at this moment: {df.shape}")
        model, preds, model_type, diagnostics = self._fit_and_diagnose(X, y)
    
        # Final report dictionary
        report = {
            "model_type": model_type,
            "diagnostics": diagnostics,
            "date_features": self.date_features_,
            "unique_categorical_keys": self.unique_categorical_keys_,
            "unique_numerical_ids": self.unique_numerical_ids_,
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "leakage_flags": self.leakage_flags,
            "imputation_values": self.imputation_values_,
        }
    
        if self.verbose: print("✅ ModelDataProfiler run complete.")
        return report, model, X, y, df
    
    def summarize_profile(self, detailed=False):
        """
        Summarize key findings from ModelDataProfiler after .profile() is run.
    
        Parameters
        ----------
        detailed : bool, default=False
            If True, prints extended lists (e.g., all leakage flags, all IDs).
            If False, shows only top 5 per category.
        """
        print("\n" + "="*80)
        print("📊 MODEL DATA PROFILER SUMMARY")
        print("="*80)
    
        # --- Model Overview ---
        model_type = getattr(self, "model_", None)
        if model_type is not None:
            model_name = type(self.model_).__name__
        else:
            model_name = "Not fitted"
    
        print(f"🧮 Baseline Model: {model_name}")
        if hasattr(self, "diagnostics_") and self.diagnostics_:
            rec = self.diagnostics_.get("recommendation", None)
            if rec:
                print(f"🔎 Recommendation: {rec}")
        print("-"*80)
    
        # --- Feature Composition ---
        print("📁 Feature Composition")
        print(f"   • Numerical features: {len(getattr(self, 'numerical_features', []))}")
        print(f"   • Categorical features: {len(getattr(self, 'categorical_features', []))}")
        print(f"   • Date features: {len(getattr(self, 'date_features_', []))}")
        print(f"   • Missing indicators: {sum(col.endswith('_was_missing') for col in getattr(self, 'numerical_features', []))}")
        print("-"*80)
    
        # --- High Cardinality / Unique IDs ---
        print("🔑 Unique Identifiers and High Cardinality Checks")
        cat_keys = getattr(self, "unique_categorical_keys_", [])
        num_ids = getattr(self, "unique_numerical_ids_", [])
        high_card = getattr(self, "high_cardinality_features_", [])
    
        def summarize_list(lst, label):
            if not lst:
                print(f"   • No {label} detected ✅")
            else:
                display = lst if detailed else lst[:5]
                more = f"... (+{len(lst)-5} more)" if len(lst) > 5 and not detailed else ""
                print(f"   • {label}: {display} {more}")
    
        summarize_list(cat_keys, "categorical keys")
        summarize_list(num_ids, "numeric IDs")
        summarize_list(high_card, "high-cardinality categorical features")
        print("-"*80)
    
        # --- Imputation Summary ---
        print("🧩 Missing Value Handling")
        imputations = getattr(self, "imputation_values_", {})
        if not imputations:
            print("   • No missing values detected ✅")
        else:
            print(f"   • Imputed {len(imputations)} columns:")
            display = list(imputations.keys()) if detailed else list(imputations.keys())[:5]
            more = f"... (+{len(imputations)-5} more)" if len(imputations) > 5 and not detailed else ""
            for c in display:
                val = imputations[c]
                print(f"      - {c} → {val}")
            if more:
                print(f"      {more}")
        print("-"*80)
    
        # --- Leakage Summary ---
        print("🕵️ Leakage Detection")
        leaks = getattr(self, "leakage_flags", {})
        if not leaks:
            print("   • No target leakage detected ✅")
        else:
            print(f"   ⚠️ {len(leaks)} suspicious features detected:")
            for i, (feat, issues) in enumerate(leaks.items()):
                if not detailed and i >= 5:
                    print(f"   ... (+{len(leaks)-5} more)")
                    break
                msg = issues[0]['message'] if isinstance(issues, list) and issues else "Potential proxy/leakage"
                print(f"      - {feat}: {msg}")
        print("-"*80)
    
        # --- Diagnostic Metrics ---
        print("📈 Diagnostic Metrics")
        diag = getattr(self, "diagnostics_", {})
        if diag:
            if "autocorrelation" in diag:
                dw = diag["autocorrelation"].get("durbin_watson", None)
                if dw:
                    dw_flag = "✅ OK" if 1.5 < dw < 2.5 else "⚠️ Potential dependency"
                    print(f"   • Durbin–Watson: {dw:.2f} ({dw_flag})")
            if "multicollinearity" in diag:
                vif = pd.DataFrame(diag["multicollinearity"]["vif"])
                high_vif = vif[vif["VIF"] > 10]
                print(f"   • High VIF features (>10): {len(high_vif)}")
            if "performance" in diag:
                perf = diag["performance"]
                if "log_loss" in perf:
                    print(f"   • Log Loss: {perf['log_loss']:.4f}")
                elif "R2" in perf:
                    print(f"   • R²: {perf['R2']:.3f}")
        else:
            print("   • No diagnostics available.")
        print("-"*80)
    
        print("✅ Summary complete.")
        print("="*80 + "\n")
    
    def to_report_dict(self):
        """
        Return a structured dictionary summarizing profiling results.
    
        This output is JSON-serializable and designed for programmatic use
        (dashboards, ML pipelines, logging, etc.)
        """
        report = {}
    
        # --- Model Metadata ---
        report["model"] = {
            "name": type(getattr(self, "model_", None)).__name__ if hasattr(self, "model_") else None
        }
    
        # --- Feature Overview ---
        report["features"] = {
            "n_numerical": len(getattr(self, "numerical_features", [])),
            "n_categorical": len(getattr(self, "categorical_features", [])),
            "n_date_features": len(getattr(self, "date_features_", [])),
            "n_missing_indicators": sum(col.endswith("_was_missing") for col in getattr(self, "numerical_features", [])),
            "numerical": getattr(self, "numerical_features", []),
            "categorical": getattr(self, "categorical_features", []),
            "date_features": getattr(self, "date_features_", [])
        }
    
        # --- Identifiers & High Cardinality ---
        report["identifier_analysis"] = {
            "categorical_keys": getattr(self, "unique_categorical_keys_", []),
            "numeric_ids": getattr(self, "unique_numerical_ids_", []),
            "high_cardinality_features": getattr(self, "high_cardinality_features_", [])
        }
    
        # --- Imputation ---
        imputations = getattr(self, "imputation_values_", {})
        report["imputation"] = {
            "n_imputed": len(imputations),
            "imputed_columns": {col: str(val) for col, val in imputations.items()}
        }
    
        # --- Leakage Detection ---
        leaks = getattr(self, "leakage_flags", {})
        report["leakage"] = {
            "n_suspicious": len(leaks),
            "flagged_features": list(leaks.keys()),
            "details": leaks
        }
    
        # --- Diagnostics ---
        diagnostics = getattr(self, "results", {})
        diag_section = {}
    
        if "autocorrelation" in diagnostics:
            dw = diagnostics["autocorrelation"].get("durbin_watson", None)
            diag_section["durbin_watson"] = {
                "value": dw,
                "status": "ok" if dw and 1.5 < dw < 2.5 else "warning"
            }
    
        if "multicollinearity" in diagnostics:
            vif = diagnostics["multicollinearity"].get("vif", [])
            high_vif = [f["feature"] for f in vif if isinstance(f["VIF"], (int, float)) and f["VIF"] > 10]
            diag_section["multicollinearity"] = {
                "high_vif_count": len(high_vif),
                "high_vif_features": high_vif
            }
    
        if "performance" in diagnostics:
            diag_section["performance"] = diagnostics["performance"]
    
        diag_section["recommendation"] = self.diagnostics_.get("recommendation_type", None)
        report["diagnostics"] = diag_section
    
        # --- Summary Metadata ---
        report["meta"] = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "class_name": type(self).__name__,
            "version": getattr(self, "__version__", "1.0"),
            "n_samples": getattr(self, "n_samples_", None),
            "n_features": len(report["features"]["numerical"]) + len(report["features"]["categorical"])
        }
    
        return report



    def save_report(self, output_dir: str = "reports", prefix: str = "model_profile", include_text: bool = True):
        """
        Save the profiler's results as both JSON and (optionally) text summaries.
    
        Uses a safe JSON serializer that handles pandas Timestamps and numpy types.
        """
        os.makedirs(output_dir, exist_ok=True)
    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f"{prefix}_{timestamp}.json")
        txt_path = os.path.join(output_dir, f"{prefix}_{timestamp}.txt")
    
        # Generate reports
        report_dict = self.to_report_dict()
    
        # --- Save JSON report ---
        with open(json_path, "w", encoding="utf-8") as f_json:
            json.dump(report_dict, f_json, indent=2, ensure_ascii=False, default=_json_serializer)
    
        # --- Save text summary (optional) ---
        if include_text:
            # summarize_profile prints to stdout; we provide a compact textual summary file
            summary_lines = []
            summary_lines.append("ModelDataProfiler Summary")
            summary_lines.append("========================")
            if hasattr(self, "diagnostics_") and self.diagnostics_:
                summary_lines.append(f"Recommendation: {self.diagnostics_.get('recommendation', '')}")
            summary_text = "\n".join(summary_lines)
            with open(txt_path, "w", encoding="utf-8") as f_txt:
                f_txt.write(summary_text)
    
        if self.verbose:
            print(f"📦 Reports saved:")
            print(f"   - JSON report: {json_path}")
            if include_text:
                print(f"   - Text summary: {txt_path}")
    
        return {"json": json_path, "text": txt_path if include_text else None}


class AutoFeatureInspectorNNI:
    """
    Automatically:
    1. Detects and fixes datatype mismatches (e.g. "36 months", "7.9%").
    2. Automatically infers categorical/numerical features.
    3. Determines task type (classification/regression) from target.
    4. Selects features via NNI FeatureGradientSelector.
    5. Tests data assumptions and recommends 'linear' or 'non-linear' model.
    """

    def __init__(self, data: pd.DataFrame, target: str, verbose=True):
        self.data = data.copy()
        self.target = target
        self.verbose = verbose

        self.fixed_data = None
        self.anomalies = {}
        self.categorical_features = []
        self.numerical_features = []
        self.selected_features = []
        self.model_type = None
        self.task_type = None  # 'regression' or 'classification'

    # ---------------------------- STEP 1 ---------------------------- #
    def _infer_and_fix_types(self):
        df = self.data.copy()
        anomalies = {}

        for col in df.columns:
            if col == self.target:
                continue
            anomalies[col] = []

            # Convert percentages
            if df[col].astype(str).str.endswith('%').any():
                anomalies[col].append("Contains % symbols – converted to float.")
                df[col] = df[col].astype(str).str.replace('%', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Convert "36 months" etc.
            elif df[col].astype(str).str.contains(r'\d+\s*(month|months|day|days|year|years)', case=False, regex=True).any():
                anomalies[col].append("Contains units (months, days, years) – numeric extracted.")
                df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)')[0]
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Convert numeric strings
            elif df[col].dtype == 'object':
                converted = pd.to_numeric(df[col], errors='coerce')
                if converted.notnull().sum() / len(df[col]) > 0.9:
                    anomalies[col].append("Stored as string but mostly numeric – converted.")
                    df[col] = converted

            # High-null flag
            if df[col].isnull().mean() > 0.5:
                anomalies[col].append("High null ratio (>50%).")

        # Infer categorical/numerical features
        categorical, numerical = [], []
        for col in df.columns:
            if col == self.target:
                continue
            unique_ratio = df[col].nunique() / len(df[col])
            if df[col].dtype == 'object' or unique_ratio < 0.05:
                categorical.append(col)
            else:
                numerical.append(col)

        # Infer target type
        if df[self.target].nunique() <= 10 and df[self.target].dtype != 'float':
            self.task_type = "classification"
        else:
            self.task_type = "regression"

        self.fixed_data = df
        self.anomalies = {k: v for k, v in anomalies.items() if v}
        self.categorical_features = categorical
        self.numerical_features = numerical

        if self.verbose:
            print(f"✔ Type inference complete: {len(categorical)} categorical, {len(numerical)} numerical features.")
            print(f"✔ Task type inferred: {self.task_type}")
            print(f"✔ {len(self.anomalies)} anomalies found.")

    # ---------------------------- STEP 2 ---------------------------- #
    def _select_features_with_nni(self):
        df = self.fixed_data.copy().dropna(subset=[self.target])
        X = df.drop(columns=[self.target])
        y = df[self.target]

        # Encode categoricals
        for col in self.categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        X = X.fillna(X.median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if self.task_type == "classification":
            print(f"type of task: {self.task_type}")
            selector = FeatureGradientSelector(classification=True)
        else:
            print(f"type of task: {self.task_type}")
            selector = FeatureGradientSelector(classification=False)

        selector.fit(X_scaled, y)
        
        # Initialize selector without arguments for compatibility.
        selector = FeatureGradientSelector()
        
        # The classification flag is passed in the fit method for some versions.
        is_classification = self.task_type == "classification"
        selector.fit(X_scaled, y, classification=is_classification)
        
        selected_mask = selector.get_support()
        self.selected_features = X.columns[selected_mask].tolist()

        if self.verbose:
            print(f"✔ Selected {len(self.selected_features)} features using NNI GradientFeatureSelector.")

    # ---------------------------- STEP 3 ---------------------------- #
    def _test_model_assumptions(self):
        df = self.fixed_data.dropna(subset=[self.target]).copy()
        if not self.selected_features:
            self.selected_features = self.numerical_features

        X = df[self.selected_features].fillna(df[self.selected_features].median())
        y = df[self.target]

        # Encode target if classification
        if self.task_type == "classification":
            le = LabelEncoder()
            y = le.fit_transform(y)

        if self.task_type == "regression":
            model = LinearRegression()
            model.fit(X, y)
            preds = model.predict(X)
            residuals = y - preds

            normal_p = shapiro(residuals)[1] if len(residuals) > 3 else 0
            spearman_corr = abs(spearmanr(preds, y)[0])
            lev_p = levene(preds, residuals)[1] if len(residuals) > 3 else 0

            linearity_score = (normal_p > 0.05) + (spearman_corr > 0.8) + (lev_p > 0.05)
            self.model_type = "linear" if linearity_score >= 2 else "non-linear"

        else:  # classification
            model = LogisticRegression(max_iter=200)
            model.fit(X, y)
            preds = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, preds)

            # Rule of thumb: high AUC with logistic => linear separability
            self.model_type = "linear" if auc > 0.8 else "non-linear"

        if self.verbose:
            print(f"✔ Model recommendation → {self.model_type} ({self.task_type})")

    # ---------------------------- PIPELINE ---------------------------- #
    def run(self):
        self._infer_and_fix_types()
        self._select_features_with_nni()
        self._test_model_assumptions()
        return {
            "anomalies": self.anomalies,
            "categorical_features": self.categorical_features,
            "numerical_features": self.numerical_features,
            "selected_features": self.selected_features,
            "task_type": self.task_type,
            "model_recommendation": self.model_type,
        }