# will identify, based on data, which is the most defining bit here.
# lets study the possibility of using the evidently pack. 
# only making pairwise comparisons ignores if a linear combination of features can predict a target. 

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
import warnings
from pathlib import Path
from typing import Union, Dict, Any, List
warnings.filterwarnings("ignore")

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

    def _load_data(self, data_source: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
        """Loads data from various sources into a pandas DataFrame."""
        if isinstance(data_source, pd.DataFrame):
            return data_source.copy()

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

    def _calculate_correlations(self, numerical_cols: List[str]):
        """Calculates Pearson and Kendall correlations for numerical columns."""
        numerical_df = self.df[numerical_cols]
        if numerical_df.empty:
            self.profile['correlations'] = {'pearson': {}, 'kendall': {}}
            return

        pearson_corr = numerical_df.corr(method='pearson')
        kendall_corr = numerical_df.corr(method='kendall')

        self.profile['correlations'] = {
            'pearson': pearson_corr.to_dict(),
            'kendall': kendall_corr.to_dict()
        }

    def _check_normality(self, numerical_cols: List[str], p_value_threshold: float = 0.05):
        """
        Performs Shapiro-Wilk test for normality on numerical columns.
        A feature's distribution is considered 'parametric' (normal) if the p-value
        is above the threshold.
        """
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

    def _describe_domain(self, numerical_cols: List[str], categorical_cols: List[str]):
        """Describes the domain for each feature."""
        domain_info = {}
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

    def run(self) -> Dict[str, Any]:
        """
        Runs the full data profiling analysis.

        Returns:
            Dict[str, Any]: A dictionary containing the full data profile.
        """
        numerical_cols, categorical_cols = self._get_feature_types()
        self._calculate_correlations(numerical_cols)
        self._check_normality(numerical_cols)
        self._describe_domain(numerical_cols, categorical_cols)
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
    def __init__(self, data, categorical_features, numerical_features, target, categorical_features_order=None):
        self.data = self._load_data(data)
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.target = target
        self.cat_order = categorical_features_order
        self.results = {}

    def _load_data(self, data):
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, str):
            if data.endswith('.csv'):
                return pd.read_csv(data)
            elif data.endswith('.parquet'):
                return pd.read_parquet(data)
            else:
                raise ValueError("File format not supported: use CSV or Parquet")
        else:
            raise ValueError("Data must be a DataFrame or file path")

    def _test_normality(self, series):
        """Returns Shapiro, D’Agostino and Anderson-Darling results."""
        shapiro_p = stats.shapiro(series.sample(min(5000, len(series))))[1]
        dagostino_p = stats.normaltest(series)[1]
        ad_result = stats.anderson(series)
        ad_signif = np.mean(ad_result.statistic < ad_result.critical_values)
        return {
            'shapiro_p': shapiro_p,
            'dagostino_p': dagostino_p,
            'anderson_reject': ad_signif < 0.5
        }

    def _compute_vif(self, X):
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                           for i in range(X.shape[1])]
        return vif_data

    def _test_homoscedasticity(self, X, y):
        """Breusch–Pagan test for regression-style homoscedasticity."""
        import statsmodels.api as sm
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        test = het_breuschpagan(model.resid, X_const)
        return dict(zip(['LM stat', 'LM pvalue', 'F stat', 'F pvalue'], test))

    def _nonlinearity_test(self, X, y):
        """Compare linear correlation vs nonlinear (mutual information)."""
        corr = np.abs([np.corrcoef(X[col], y)[0, 1] for col in X.columns])
        if np.issubdtype(y.dtype, np.number):
            mi = mutual_info_regression(X, y)
        else:
            le = LabelEncoder()
            mi = mutual_info_classif(X, le.fit_transform(y))
        ratio = mi / (np.abs(corr) + 1e-8)
        nonlinear_features = X.columns[ratio > 2].tolist()
        return nonlinear_features

    def profile(self):
        df = self.data.copy()

        # --- Normality Tests ---
        normality_results = {}
        for col in self.numerical_features:
            try:
                normality_results[col] = self._test_normality(df[col].dropna())
            except Exception as e:
                normality_results[col] = {"error": str(e)}

        # --- Homoscedasticity ---
        if np.issubdtype(df[self.target].dtype, np.number):
            homo = self._test_homoscedasticity(df[self.numerical_features], df[self.target])
        else:
            homo = None

        # --- Multicollinearity ---
        vif = self._compute_vif(df[self.numerical_features])

        # --- Nonlinearity ---
        nonlinear = self._nonlinearity_test(df[self.numerical_features], df[self.target])

        # --- Outlier summary ---
        outlier_ratio = {}
        for col in self.numerical_features:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_ratio[col] = np.mean(z_scores > 3)

        # --- Summary ---
        self.results = {
            "normality": normality_results,
            "homoscedasticity": homo,
            "multicollinearity_vif": vif,
            "nonlinearity_features": nonlinear,
            "outlier_ratio": outlier_ratio
        }

        return self._summarize()

    def _summarize(self):
        norm_fail = sum(
            [res['shapiro_p'] < 0.05 for res in self.results['normality'].values() if 'shapiro_p' in res])
        high_vif = (self.results['multicollinearity_vif']['VIF'] > 5).sum()
        nonlinear_count = len(self.results['nonlinearity_features'])

        recommendation = "Parametric models OK"
        if norm_fail > len(self.numerical_features) / 2 or nonlinear_count > 0 or high_vif > 2:
            recommendation = "Prefer non-parametric models (e.g., tree-based, ensemble, kernel methods)"

        return {
            "summary": {
                "normality_fail_count": norm_fail,
                "high_vif_count": int(high_vif),
                "nonlinear_features": nonlinear_count,
                "recommendation": recommendation
            },
            **self.results
        }
