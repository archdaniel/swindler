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
warnings.filterwarnings("ignore")

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
                sample = data.sample(n=min(len(data), 4999), random_state=1)
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
    def __init__(self, data, categorical_features, numerical_features, target, categorical_features_order=None, verbose=True):
        self.verbose = verbose
        self.data = self._load_data(data, verbose=verbose)
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.target = target
        self.cat_order = categorical_features_order
        self.model = None
        self.results = {}

    def _load_data(self, data, verbose: bool = True):
        if isinstance(data, pd.DataFrame):
            if verbose: print("Loading data for ModelDataProfiler from DataFrame... really...")
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
    def profile_data_encoding(self, fix=True):
        """
        Detects, describes, and optionally fixes encoding/storage anomalies in dataset features.

        Parameters
        ----------
        fix : bool, default=False
            If True, returns a cleaned copy of the dataframe with detected issues fixed automatically.
            Otherwise, only reports issues.

        Returns
        -------
        issues : dict
            Dictionary describing anomalies for each column.
        cleaned_df : pd.DataFrame (if fix=True)
            Cleaned copy of dataframe with numeric-like strings converted.
        """
        df = self.data.copy()
        issues = {}
        unit_pattern = re.compile(r"^\s*([-+]?\d*\.?\d+)\s*[a-zA-Z]+")  # e.g., '36 months', '5kg'
        pct_pattern = re.compile(r"^\s*([-+]?\d*\.?\d+)\s*%$")
        curr_pattern = re.compile(r"^\s*[$€]\s*([-+]?\d*\.?\d+)")

        for col in df.columns:
            series = df[col]
            col_issues = []

            # --- Detect numeric-like strings with symbols (%, $, commas)
            if series.dtype == "object":
                suspicious_mask = series.astype(str).str.contains(r'[%,$€]|[0-9]+,[0-9]+', regex=True)
                if suspicious_mask.mean() > 0.1:
                    col_issues.append(
                        f"⚠️ {round(100*suspicious_mask.mean(),1)}% of entries look numeric but contain symbols (%, $, €, commas)."
                    )

            # --- Detect numeric + unit patterns like "36 months"
            if series.dtype == "object":
                num_unit_mask = series.astype(str).str.match(unit_pattern)
                if num_unit_mask.mean() > 0.1:
                    col_issues.append(
                        f"📏 {round(100*num_unit_mask.mean(),1)}% of values appear to mix numbers and units (e.g. '36 months', '5kg')."
                    )

            # --- Detect possible numeric stored as string
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

            # --- Detect symbolic encodings
            if series.dtype == "object":
                values = series.astype(str)
                if values.str.endswith("%").mean() > 0.5:
                    col_issues.append("🧮 Appears to store percentages as text (values ending with '%').")
                if values.str.contains(r"\$|€").mean() > 0.5:
                    col_issues.append("💰 Appears to store currency values as text (contains $ or €).")

            # --- Detect numeric scale mismatches
            if pd.api.types.is_numeric_dtype(series):
                if series.max() > 10 and series.mean() < 1:
                    col_issues.append("📊 Possible % stored as 0–100 instead of 0–1.")
                elif series.max() <= 1 and series.mean() < 0.1:
                    col_issues.append("📉 Possible % stored as 0–1 instead of 0–100.") # wait...

            # --- Detect mixed types
            if series.dtype == "object":
                unique_types = series.dropna().map(type).nunique()
                if unique_types > 1:
                    col_issues.append("⚠️ Mixed data types detected (numeric + non-numeric or inconsistent formats).")

            # === OPTIONAL FIXES ===
            if fix and series.dtype == "object":
                fixed_series = series.astype(str).str.strip()

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

                # Try to coerce remaining numeric strings
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

        if fix:
            return issues, df
        else:
            return issues
            
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
        if verbose: print(f"Baseline model fitted: {model_type}.")
        return model, preds, model_type

    def _residual_analysis(self, y, preds, model_type, verbose: bool = True):
        residuals = y - preds
        if verbose: print("Performing residual analysis...")
        results = {}

        # Normality tests
        shapiro_p = stats.shapiro(residuals.sample(min(5000, len(residuals))))[1]
        ad_stat, ad_p = normal_ad(residuals)
        results['normality'] = {'shapiro_p': shapiro_p, 'anderson_darling_p': ad_p}

        # Homoscedasticity (Breusch-Pagan)
        X = sm.add_constant(preds)
        lm, lm_pvalue, f, f_pvalue = het_breuschpagan(residuals, X)
        results['homoscedasticity'] = {'LM pvalue': lm_pvalue, 'F pvalue': f_pvalue}

        # Autocorrelation (Durbin–Watson)
        dw = durbin_watson(residuals)
        results['autocorrelation'] = {'durbin_watson': dw}

        # Performance
        if model_type == "regression":
            results['performance'] = {'R2': r2_score(y, preds)}
        else:
            results['performance'] = {'log_loss': log_loss(y, preds)}

        # Visual plots
        self._plot_diagnostics(y, preds, residuals, model_type, verbose=verbose)

        # Recommendation
        if shapiro_p < 0.05 or f_pvalue < 0.05 or dw < 1.5 or dw > 2.5:
            recommendation = "⚠️ Non-parametric or robust model suggested (e.g. tree-based, Huber)"
        else:
            recommendation = "✅ Parametric assumptions reasonable"
        results['recommendation'] = recommendation
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

    def profile(self):
        if self.verbose: print("Starting ModelDataProfiler run...")
        df = self.data.copy()
 
        transformed_to_numeric = []
        date_features = []
        unique_categorical_keys = []
        unique_numerical_ids = []
        high_cardinality_threshold = 0.95
 
        # --- Pre-emptive check for numeric-like categorical features ---
        if self.verbose: print("Checking for string columns that can be converted to numeric...")
        # Iterate over a copy since we might modify the list
        for col in self.categorical_features[:]:
            # Only check object/string columns
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                if self.verbose: print(f"  - Checking column '{col}' for potential numeric conversion...")
                
                # Attempt to convert to numeric, coercing errors will turn non-numeric values into NaT/NaN
                converted_series = pd.to_numeric(df[col], errors='coerce')
                
                # If the number of nulls did not increase after conversion, it's safe to assume it's a numeric column
                original_nulls = df[col].isnull().sum()
                coerced_nulls = converted_series.isnull().sum()

                if coerced_nulls == original_nulls:
                    if self.verbose: print(f"    -> Successfully converted column '{col}' to numeric.")
                    df[col] = converted_series
                    self.numerical_features.append(col)
                    self.categorical_features.remove(col)
                    transformed_to_numeric.append(col)
                else:
                    if self.verbose: print(f"    -> Column '{col}' contains non-numeric values and will be treated as categorical.")

        # --- Check for date-like categorical features ---
        if self.verbose: print("Checking for date-like categorical features...")
        for col in self.categorical_features[:]:
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                try:
                    converted_series = pd.to_datetime(df[col], errors='coerce')
                    # If over 95% of non-null values can be converted to dates, treat as a date feature
                    if converted_series.notna().sum() / df[col].notna().sum() > 0.95:
                        if self.verbose: print(f"    -> Column '{col}' detected as a date feature.")
                        date_features.append(col)
                        self.categorical_features.remove(col)
                except Exception:
                    continue # Not a date

        # --- Check for high cardinality categorical features (potential unique keys) ---
        if self.verbose: print("Checking for high cardinality categorical features...")
        for col in self.categorical_features[:]:
            if df[col].nunique() / len(df[col]) > high_cardinality_threshold:
                if self.verbose: print(f"    -> Column '{col}' has high cardinality, treating as a unique key.")
                unique_categorical_keys.append(col)
                self.categorical_features.remove(col)

        # --- Check for high domain numerical features (potential unique identifiers) ---
        if self.verbose: print("Checking for high domain numerical features...")
        for col in self.numerical_features[:]:
            # Check for high cardinality first
            is_high_cardinality = df[col].nunique() / len(df[col]) > high_cardinality_threshold
            if is_high_cardinality:
                # Heuristic 1: Check skewness. True IDs often have low skew.
                skewness = df[col].skew()
                is_low_skew = abs(skewness) < 2.0

                # Heuristic 2: Check if the column is integer-like.
                # True IDs are rarely floats with meaningful decimal parts.
                # Check if more than 1% of values have a non-zero fractional part.
                is_float_like = (df[col].dropna() % 1 != 0).sum() / len(df[col].dropna()) > 0.01

                # Decision: An ID should have high cardinality, low skew, and NOT be float-like.
                if is_low_skew and not is_float_like:
                    if self.verbose: print(f"    -> Column '{col}' has high cardinality, low skew ({skewness:.2f}), and is integer-like. Treating as a unique identifier.")
                    unique_numerical_ids.append(col)
                    self.numerical_features.remove(col)
                else:
                    reason = []
                    if not is_low_skew: reason.append(f"is highly skewed ({skewness:.2f})")
                    if is_float_like: reason.append("contains float values")
                    if self.verbose: print(f"    -> Column '{col}' has high cardinality but {' and '.join(reason)}. Keeping as a numerical feature.")

        # --- Impute missing values ---
        if self.verbose: print("Imputing missing values for baseline modeling...")
        # Impute numerical features with the median
        # Iterate over a copy of the list because we will be modifying it
        for col in self.numerical_features[:]:
            if df[col].isnull().any():
                # Create a new binary indicator feature for missing values
                indicator_col_name = f"{col}_was_missing"
                df[indicator_col_name] = df[col].isnull().astype(int)
                self.numerical_features.append(indicator_col_name)
                if self.verbose: print(f"    -> Created missing indicator column '{indicator_col_name}' for '{col}'.")

                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                if self.verbose: print(f"    -> Imputed NaNs in numerical column '{col}' with median ({median_val}).")
        # Impute categorical features with the mode
        for col in self.categorical_features[:]:
            if df[col].isnull().any():
                # Create a new binary indicator feature for missing values
                indicator_col_name = f"{col}_was_missing"
                df[indicator_col_name] = df[col].isnull().astype(int)
                self.numerical_features.append(indicator_col_name) # this needs to be removed.
                if self.verbose: print(f"    -> Created missing indicator column '{indicator_col_name}' for '{col}'.")

                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                if self.verbose: print(f"    -> Imputed NaNs in categorical column '{col}' with mode ('{mode_val}').")
 
        # Prepare X and y
        if self.verbose: print("Preparing data for modeling (including one-hot encoding)...")
        
        # Separate numerical and categorical dataframes
        X_numerical = df[self.numerical_features].copy()
        X_categorical = pd.get_dummies(df[self.categorical_features], drop_first=True)
        print(list(X_categorical.columns))
        X = pd.concat([X_numerical, X_categorical], axis=1)

        y = df[self.target]
        if self.verbose: print(f"Data prepared. Shape of X: {X.shape}")
        
        # Fit model and analyze residuals
        self.model, preds, model_type = self._fit_baseline_model(X, y, verbose=self.verbose)
        self.results = self._residual_analysis(y, preds, model_type, verbose=self.verbose)
        if self.verbose: print("ModelDataProfiler run finished.")
        return (
            self.results, 
            self.model, 
            transformed_to_numeric, 
            date_features, 
            unique_categorical_keys, 
            unique_numerical_ids,
            self.numerical_features,
            self.categorical_features
        )




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
