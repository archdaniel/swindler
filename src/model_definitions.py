# src/model_definitions.py
# Modified to accept either a DataFrame `data` (legacy) OR arrays X/y (including sparse matrices).
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Union, Optional
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, learning_curve
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, r2_score,
    mean_squared_error, mean_absolute_error, RocCurveDisplay, ConfusionMatrixDisplay
)
from scipy import sparse as sp

seed = 505

# try optional heavy libs
try:
    import mlflow
except ImportError:
    mlflow = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import catboost as cb
except ImportError:
    cb = None

try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
    from sklearn.model_selection import cross_val_score
except Exception:
    hp = None

class ModelTrainer:
    """
    Trains, tunes, and evaluates candidate models.
    Accepts either:
      - data: full pandas DataFrame with target column, OR
      - X: ndarray or scipy.sparse matrix and y: Series/ndarray
    """
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        target_col: Optional[str] = None,
        X: Optional[Union[np.ndarray, sp.spmatrix]] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
        task_type: str = "classification",
        model_type: str = "non-linear",
        validation_strategy: str = 'train_test_split',
        hyperparam_strategy: str = 'grid_search',
        use_mlflow: bool = True,
        local_model_path: str = "best_model.joblib",
        mlflow_experiment_name: str = "Default Experiment",
        test_size: float = 0.2,
        cv_folds: int = 5,
        random_state: int = 42,
        verbose: bool = True
    ):
        self.task_type = task_type
        self.model_type = model_type
        self.validation_strategy = validation_strategy
        self.hyperparam_strategy = hyperparam_strategy
        self.use_mlflow = use_mlflow
        self.local_model_path = local_model_path
        self.mlflow_experiment_name = mlflow_experiment_name
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state if random_state != seed else seed
        self.verbose = verbose

        # Input handling: either data OR X,y must be provided
        if data is not None:
            if target_col is None:
                raise ValueError("target_col required when passing `data` DataFrame")
            self.data = data
            self.target_col = target_col
            self.features_used_: List[str] = [col for col in data.columns if col != target_col]
            self.X = data[self.features_used_]
            self.y = data[self.target_col]
        elif X is not None and y is not None:
            # X can be sparse matrix or ndarray
            self.data = None
            self.target_col = None
            self.X = X
            self.y = np.array(y).ravel()
            if feature_names:
                self.features_used_ = feature_names
            else:
                # create generic names for sparse/array inputs (one per column)
                n_feat = int(X.shape[1]) if hasattr(X, "shape") else None
                if n_feat is None:
                    raise ValueError("Cannot infer number of features from X; provide feature_names.")
                self.features_used_ = [f"f{i}" for i in range(n_feat)]
        else:
            raise ValueError("Either `data` (DataFrame) or `X` and `y` must be provided to ModelTrainer.")

        self.models_: Dict[str, Any] = {}
        self.results_summary_: pd.DataFrame = pd.DataFrame()
        self.best_model_name_: str = ""
        self.best_model_: Any = None
        self.best_params_: Dict[str, Any] = {}
        self.feature_importances_: pd.DataFrame = pd.DataFrame()

    def _define_candidate_models(self):
        if self.verbose: print(f"Defining candidate models for task: {self.task_type}, type: {self.model_type}")
        if self.task_type == 'classification':
            self.scoring = 'roc_auc'
            # define richer candidates (use hyperopt if available)
            if hp is not None:
                models = {
                    'LogisticRegression': (LogisticRegression(random_state=self.random_state, max_iter=1000, solver='liblinear'), {
                        'C': hp.loguniform('C', np.log(0.01), np.log(100)),
                        'penalty': hp.choice('penalty', ['l1', 'l2'])
                    })
                }
                if self.model_type == 'non-linear':
                    models['RandomForestClassifier'] = (RandomForestClassifier(random_state=self.random_state), {
                        'n_estimators': hp.choice('n_estimators', [100, 200, 300]),
                        'max_depth': hp.choice('max_depth', [10, 20, 30, None])
                    })
                    if xgb:
                        models['XGBClassifier'] = (xgb.XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='logloss'), {
                            'n_estimators': hp.choice('n_estimators', [100, 200, 500]),
                            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
                            'max_depth': hp.choice('max_depth', [3, 5, 7])
                        })
                    if lgb:
                        models['LGBMClassifier'] = (lgb.LGBMClassifier(random_state=self.random_state), {
                            'n_estimators': hp.choice('n_estimators', [100, 200, 500]),
                            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
                            'num_leaves': hp.choice('num_leaves', [20, 31, 40, 50])
                        })
                    if cb:
                        models['CatBoostClassifier'] = (cb.CatBoostClassifier(random_state=self.random_state, verbose=0), {
                            'iterations': hp.choice('iterations', [100, 200, 500]),
                            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
                            'depth': hp.choice('depth', [4, 6, 8])
                        })
            else:
                # fallback to safe grids (no hyperopt dependency)
                models = {
                    'LogisticRegression': (LogisticRegression(random_state=self.random_state, max_iter=1000, solver='liblinear'), {
                        'C': [0.01, 0.1, 1.0],
                        'penalty': ['l2']
                    })
                }
                if self.model_type == 'non-linear':
                    models['RandomForestClassifier'] = (RandomForestClassifier(random_state=self.random_state), {
                        'n_estimators': [100, 200],
                        'max_depth': [10, None]
                    })
                    if xgb:
                        models['XGBClassifier'] = (xgb.XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='logloss'), {
                            'n_estimators': [100, 200],
                            'learning_rate': [0.05, 0.1],
                            'max_depth': [3, 5]
                        })
                    if lgb:
                        models['LGBMClassifier'] = (lgb.LGBMClassifier(random_state=self.random_state), {
                            'n_estimators': [100, 200],
                            'learning_rate': [0.05, 0.1]
                        })
                    if cb:
                        models['CatBoostClassifier'] = (cb.CatBoostClassifier(random_state=self.random_state, verbose=0), {
                            'iterations': [100, 200],
                            'learning_rate': [0.05, 0.1]
                        })
        else:
            self.scoring = 'r2'
            if hp is not None:
                models = {
                    'Ridge': (Ridge(random_state=self.random_state), {
                        'alpha': hp.loguniform('alpha', np.log(0.1), np.log(10.0))
                    })
                }
                if self.model_type == 'non-linear':
                    models['RandomForestRegressor'] = (RandomForestRegressor(random_state=self.random_state), {
                        'n_estimators': hp.choice('n_estimators', [100, 200, 300]),
                        'max_depth': hp.choice('max_depth', [10, 20, 30, None])
                    })
                    if xgb:
                        models['XGBRegressor'] = (xgb.XGBRegressor(random_state=self.random_state), {
                            'n_estimators': hp.choice('n_estimators', [100, 200, 500]),
                            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
                            'max_depth': hp.choice('max_depth', [3, 5, 7])
                        })
                    if lgb:
                        models['LGBMRegressor'] = (lgb.LGBMRegressor(random_state=self.random_state), {
                            'n_estimators': hp.choice('n_estimators', [100, 200, 500]),
                            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
                            'num_leaves': hp.choice('num_leaves', [20, 31, 40, 50])
                        })
                    if cb:
                        models['CatBoostRegressor'] = (cb.CatBoostRegressor(random_state=self.random_state, verbose=0), {
                            'iterations': hp.choice('iterations', [100, 200, 500]),
                            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
                            'depth': hp.choice('depth', [4, 6, 8])
                        })
            else:
                models = {
                    'Ridge': (Ridge(random_state=self.random_state), {
                        'alpha': [1.0]
                    })
                }
                if self.model_type == 'non-linear':
                    models['RandomForestRegressor'] = (RandomForestRegressor(random_state=self.random_state), {
                        'n_estimators': [100],
                        'max_depth': [10, None]
                    })
        self.models_ = models

    def _train_and_tune(self):
        """Trains and tunes all candidate models."""
        # train/val/test split to support early stopping
        if sp.issparse(self.X) or isinstance(self.X, np.ndarray):
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=self.random_state,
                stratify=(self.y if self.task_type == 'classification' and len(np.unique(self.y))>1 else None)
            )
        else:
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=self.random_state,
                stratify=(self.y if self.task_type == 'classification' and len(np.unique(self.y))>1 else None)
            )

        # carve out a small validation set for early stopping
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval, test_size=0.1, random_state=self.random_state,
                stratify=(y_trainval if self.task_type == 'classification' and len(np.unique(y_trainval))>1 else None)
            )
        except Exception:
            X_train, X_val, y_train, y_val = X_trainval, None, y_trainval, None

        results = []
        for name, (model, params) in self.models_.items():
            if self.verbose: print(f"--- Training {name} ---")
            best_estimator = None
            best_params = {}

            # prepare model-specific fit params for early stopping
            fit_params = {}
            supports_early = False
            if name.startswith("XGB") and xgb and X_val is not None:
                supports_early = True
                fit_params = {'eval_set': [(X_val, y_val)], 'early_stopping_rounds': 10, 'verbose': False}
            elif name.startswith("LGBM") and lgb and X_val is not None:
                supports_early = True
                fit_params = {'eval_set': [(X_val, y_val)], 'early_stopping_rounds': 10, 'verbose': False}
            elif name.startswith("CatBoost") and cb and X_val is not None:
                supports_early = True
                fit_params = {'eval_set': (X_val, y_val), 'early_stopping_rounds': 10, 'verbose': False}

            # If using hyperopt space, run hyperopt (slow) otherwise GridSearchCV
            try:
                if params and hp is not None and any(getattr(v, 'name', None) is not None for v in (params.values() if isinstance(params, dict) else [])):
                    # hyperopt flow
                    def objective(hyperparams):
                        if 'iterations' in hyperparams:
                            hyperparams['iterations'] = int(hyperparams['iterations'])
                        clf = model.set_params(**hyperparams)
                        try:
                            score = cross_val_score(clf, X_train, y_train, cv=self.cv_folds, scoring=self.scoring, n_jobs=-1).mean()
                        except Exception:
                            score = 0.0
                        return {'loss': -score, 'status': STATUS_OK}
                    trials = Trials()
                    best_h = fmin(fn=objective, space=params, algo=tpe.suggest, max_evals=10, trials=trials, rstate=np.random.default_rng(self.random_state))
                    best_params = space_eval(params, best_h)
                    if 'iterations' in best_params:
                        best_params['iterations'] = int(best_params['iterations'])
                    best_estimator = model.set_params(**best_params).fit(X_train, y_train, **(fit_params if supports_early else {}))
                elif params:
                    # fallback to GridSearchCV.
                    search_cv = GridSearchCV(model, params, cv=self.cv_folds, scoring=self.scoring, n_jobs=-1)
                    if supports_early and X_val is not None:
                        # pass fit_params if estimator supports eval_set
                        search_cv.fit(X_train, y_train, **fit_params)
                    else:
                        search_cv.fit(X_train, y_train)
                    best_estimator = search_cv.best_estimator_
                    best_params = search_cv.best_params_
                else:
                    # no hyperparams to tune
                    if supports_early and X_val is not None:
                        best_estimator = model.fit(X_train, y_train, **fit_params)
                    else:
                        best_estimator = model.fit(X_train, y_train)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Training/search failed for {name} with error: {e}. Falling back to direct fit.")
                try:
                    best_estimator = model.fit(X_train, y_train)
                except Exception as e2:
                    if self.verbose:
                        print(f"Direct fit also failed for {name}: {e2}")
                    continue

            # evaluate on test set
            try:
                y_pred = best_estimator.predict(X_test)
            except Exception:
                y_pred = best_estimator.predict(X_test)

            model_results = {'model': name, 'best_params': best_params}
            metrics_to_log = {}
            if self.task_type == 'classification':
                try:
                    y_proba = best_estimator.predict_proba(X_test)[:, 1]
                except Exception:
                    y_proba = None
                metrics_to_log['accuracy'] = accuracy_score(y_test, y_pred)
                metrics_to_log['f1'] = f1_score(y_test, y_pred, zero_division=0)
                metrics_to_log['precision'] = precision_score(y_test, y_pred, zero_division=0)
                metrics_to_log['recall'] = recall_score(y_test, y_pred, zero_division=0)
                metrics_to_log['roc_auc'] = float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None
            else:
                metrics_to_log['r2'] = r2_score(y_test, y_pred)
                metrics_to_log['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))

            model_results.update(metrics_to_log)
            results.append(model_results)
            self.models_[name] = (best_estimator, best_params)

        self.results_summary_ = pd.DataFrame(results)

    def _evaluate_and_select_best(self):
        if self.results_summary_.empty:
            raise RuntimeError("Models have not been trained yet. Run _train_and_tune() first.")
        if self.task_type == 'classification':
            if 'roc_auc' in self.results_summary_.columns and self.results_summary_['roc_auc'].notna().any():
                self.best_model_name_ = self.results_summary_.loc[self.results_summary_['roc_auc'].idxmax()]['model']
            else:
                self.best_model_name_ = self.results_summary_.iloc[0]['model']
        else:
            self.best_model_name_ = self.results_summary_.loc[self.results_summary_['r2'].idxmax()]['model']
        self.best_model_, self.best_params_ = self.models_[self.best_model_name_]

        if self.verbose: print(f"\n🏆 Best Model Selected: {self.best_model_name_}")

        # Build feature importances safely, guarding against length mismatches.
        try:
            if hasattr(self.best_model_, 'feature_importances_'):
                importances = np.array(self.best_model_.feature_importances_).ravel()
                # align feature names length and importances length
                if len(importances) == len(self.features_used_):
                    features = self.features_used_
                else:
                    # fallback: synthesize feature names if sizes differ
                    features = [f"f{i}" for i in range(len(importances))]
                self.feature_importances_ = pd.DataFrame({
                    'feature': features,
                    'importance': importances
                }).sort_values('importance', ascending=False)
            elif hasattr(self.best_model_, 'coef_'):
                coef = np.array(self.best_model_.coef_)
                # flatten in a sensible way: if multi-class, sum absolute coefs across classes
                if coef.ndim == 1:
                    importances = np.abs(coef)
                else:
                    importances = np.abs(coef).sum(axis=0)
                importances = importances.ravel()
                if len(importances) == len(self.features_used_):
                    features = self.features_used_
                else:
                    features = [f"f{i}" for i in range(len(importances))]
                self.feature_importances_ = pd.DataFrame({
                    'feature': features,
                    'importance': importances
                }).sort_values('importance', ascending=False)
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Could not compute feature importances: {e}")
            self.feature_importances_ = pd.DataFrame()

    def _plot_results(self):
        if self.results_summary_.empty:
            print("No results to plot.")
            return
        melted_results = self.results_summary_.melt(id_vars='model', var_name='metric', value_name='score')
        metrics_to_plot = [m for m in melted_results['metric'].unique() if m != 'best_params']
        melted_results = melted_results[melted_results['metric'].isin(metrics_to_plot)]

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(data=melted_results, x='score', y='model', hue='metric', orient='h', ax=ax)
        plt.show()

    def run(self):
        if self.use_mlflow and mlflow:
            mlflow.set_experiment(self.mlflow_experiment_name)
        elif self.use_mlflow:
            if self.verbose: print("MLflow not found. Skipping logging.")
        self._define_candidate_models()
        self._train_and_tune()
        self._evaluate_and_select_best()
        self._plot_results()
        if not self.use_mlflow:
            try:
                self.save_best_model(self.local_model_path)
            except Exception:
                pass
        return self.best_model_, self.feature_importances_, self.features_used_

    def save_best_model(self, path: str, format: str = 'joblib'):
        if not self.best_model_:
            raise RuntimeError("No best model available to save. Run the trainer first.")
        if format == 'joblib':
            joblib.dump(self.best_model_, path)
            if self.verbose: print(f"Best model ({self.best_model_name_}) saved to {path}")
        else:
            raise ValueError(f"Unsupported save format: {format}. Use 'joblib'.")