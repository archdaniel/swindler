# based on auto_definitions.py, will select which model type to be used.
# study the possibility of using mflow for logging.
# model definitions and their different variations.

from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Union
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
    r2_score, mean_squared_error, mean_absolute_error,
    roc_curve, confusion_matrix
)

seed = 505
# --- MLflow Imports ---

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
except ImportError:
    hp = None

class ModelTrainer:
    """
    Trains, tunes, and evaluates a pool of candidate models based on data
    characteristics and returns the best performer.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        task_type: str,
        model_type: str,
        validation_strategy: str = 'train_test_split',
        hyperparam_strategy: str = 'grid_search', # 'grid_search', 'random_search', 'hyperopt'
        use_mlflow: bool = True,
        local_model_path: str = "best_model.joblib",
        mlflow_experiment_name: str = "Default Experiment",
        test_size: float = 0.2,
        cv_folds: int = 5,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Args:
            data (pd.DataFrame): Preprocessed dataframe with features and target.
            target_col (str): Name of the target column.
            task_type (str): 'classification' or 'regression'.
            model_type (str): 'linear' or 'non-linear'.
            validation_strategy (str): 'train_test_split' or 'cross_validation'.
            hyperparam_strategy (str): 'grid_search', 'random_search', or 'hyperopt'.
            use_mlflow (bool): If True, use MLflow for experiment tracking.
            local_model_path (str): Path to save the best model if `use_mlflow` is False.
            mlflow_experiment_name (str): The name of the MLflow experiment to log runs to.
            test_size (float): Proportion of the dataset to include in the test split.
            cv_folds (int): Number of folds for cross-validation.
            random_state (int): Seed for reproducibility.
            verbose (bool): If True, prints progress information.
        """
        self.data = data
        self.target_col = target_col
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

        self.features_used_: List[str] = [col for col in data.columns if col != target_col]
        self.X = self.data[self.features_used_]
        self.y = self.data[self.target_col]

        self.models_: Dict[str, Any] = {}
        self.results_summary_: pd.DataFrame = pd.DataFrame()
        self.best_model_name_: str = ""
        self.best_model_: Any = None
        self.best_params_: Dict[str, Any] = {}
        self.feature_importances_: pd.DataFrame = pd.DataFrame()

    def _define_candidate_models(self):
        """Defines candidate models and their hyperparameter grids."""
        if self.verbose: print(f"Defining candidate models for task: {self.task_type}, type: {self.model_type}")
        
        if self.task_type == 'classification':
            self.scoring = 'roc_auc'
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
        elif self.task_type == 'regression':
            self.scoring = 'r2'
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
            raise ValueError(f"Unsupported task_type: {self.task_type}")

        self.models_ = models

    def _train_and_tune(self):
        """Trains and tunes all candidate models."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=(self.y if self.task_type == 'classification' else None)
        )
        
        results = []
        for name, (model, params) in self.models_.items():
            if self.verbose: print(f"--- Training {name} ---")

            # --- MLflow: Start a new run for each model ---
            if self.use_mlflow and mlflow:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"{name}_{timestamp}"
                mlflow.start_run(run_name=run_name)
                mlflow.log_param("model_name", name)
                mlflow.log_param("hyperparam_strategy", self.hyperparam_strategy)

            best_estimator = None
            best_params = {}
            if params:  # Only run search if there are params to tune
                if self.hyperparam_strategy in ['grid_search', 'random_search']:
                    # Convert hyperopt space to grid/random search space
                    search_params = {}
                    for p_name, p_value in params.items():
                        if p_value.name == 'loguniform':
                            search_params[p_name] = np.logspace(p_value.pos_args[0].obj, p_value.pos_args[1].obj, 5)
                        elif p_value.name == 'uniform':
                            search_params[p_name] = np.linspace(p_value.pos_args[0].obj, p_value.pos_args[1].obj, 5)
                        elif p_value.name == 'choice':
                            search_params[p_name] = p_value.pos_args[1].obj

                    if self.hyperparam_strategy == 'grid_search':
                        search_cv = GridSearchCV(model, search_params, cv=self.cv_folds, scoring=self.scoring, n_jobs=-1)
                    else: # random_search
                        search_cv = RandomizedSearchCV(model, search_params, n_iter=10, cv=self.cv_folds, scoring=self.scoring, n_jobs=-1, random_state=self.random_state)

                elif self.hyperparam_strategy == 'hyperopt':
                    if hp is None:
                        raise ImportError("hyperopt is required for this strategy. Please install it with 'pip install hyperopt'.")
                    
                    def objective(hyperparams):
                        # CatBoost needs integer for iterations
                        if 'iterations' in hyperparams: hyperparams['iterations'] = int(hyperparams['iterations'])
                        
                        clf = model.set_params(**hyperparams)
                        score = cross_val_score(clf, X_train, y_train, cv=self.cv_folds, scoring=self.scoring, n_jobs=-1).mean()
                        return {'loss': -score, 'status': STATUS_OK}

                    trials = Trials()
                    best_params = fmin(fn=objective, space=params, algo=tpe.suggest, max_evals=10, trials=trials, rstate=np.random.default_rng(self.random_state))
                    best_params = space_eval(params, best_params)
                else:
                    raise ValueError(f"Unsupported hyperparam_strategy: {self.hyperparam_strategy}")

                if self.hyperparam_strategy in ['grid_search', 'random_search']:
                    search_cv.fit(X_train, y_train)
                    best_estimator = search_cv.best_estimator_
                    best_params = search_cv.best_params_
                else: # hyperopt
                    if 'iterations' in best_params: best_params['iterations'] = int(best_params['iterations'])
                    best_estimator = model.set_params(**best_params).fit(X_train, y_train)
            else: # No hyperparams to tune
                best_estimator = model.fit(X_train, y_train)

            # Evaluate on test set
            y_pred = best_estimator.predict(X_test)
            model_results = {'model': name, 'best_params': best_params}

            metrics_to_log = {}
            if self.task_type == 'classification':
                y_proba = best_estimator.predict_proba(X_test)[:, 1]
                metrics_to_log['accuracy'] = accuracy_score(y_test, y_pred)
                metrics_to_log['roc_auc'] = roc_auc_score(y_test, y_proba)
                metrics_to_log['f1'] = f1_score(y_test, y_pred)
                metrics_to_log['precision'] = precision_score(y_test, y_pred)
                metrics_to_log['recall'] = recall_score(y_test, y_pred)
            else: # Regression
                metrics_to_log['r2'] = r2_score(y_test, y_pred)
                metrics_to_log['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
                metrics_to_log['mae'] = mean_absolute_error(y_test, y_pred)

            model_results.update(metrics_to_log)
            results.append(model_results)
            self.models_[name] = (best_estimator, best_params) # Store the fitted best estimator

            # --- MLflow: Log parameters, metrics, and model ---
            if self.use_mlflow and mlflow:
                mlflow.log_params(best_params)
                mlflow.log_metrics(metrics_to_log)
                mlflow.sklearn.log_model(best_estimator, "model")
                mlflow.end_run()

        self.results_summary_ = pd.DataFrame(results)

    def _evaluate_and_select_best(self):
        """Selects the best model based on the primary scoring metric."""
        if self.results_summary_.empty:
            raise RuntimeError("Models have not been trained yet. Run _train_and_tune() first.")

        if self.task_type == 'classification':
            self.best_model_name_ = self.results_summary_.loc[self.results_summary_['roc_auc'].idxmax()]['model']
        else: # Regression
            self.best_model_name_ = self.results_summary_.loc[self.results_summary_['r2'].idxmax()]['model']

        self.best_model_, self.best_params_ = self.models_[self.best_model_name_]

        if self.verbose: print(f"\n🏆 Best Model Selected: {self.best_model_name_}")
        if self.verbose: print(f"   Best Params: {self.best_params_}")

        # Get feature importances
        if hasattr(self.best_model_, 'feature_importances_'):
            importances = self.best_model_.feature_importances_
            self.feature_importances_ = pd.DataFrame({
                'feature': self.features_used_,
                'importance': importances
            }).sort_values('importance', ascending=False)
        elif hasattr(self.best_model_, 'coef_'):
            importances = np.abs(self.best_model_.coef_.ravel())
            self.feature_importances_ = pd.DataFrame({
                'feature': self.features_used_,
                'importance': importances
            }).sort_values('importance', ascending=False)
        
        # This part is commented out as it can cause issues with nested runs.
        # # --- MLflow: Log feature importance for the best model ---
        # if self.use_mlflow and mlflow and not self.feature_importances_.empty:
        #     with mlflow.start_run(run_id=mlflow.last_active_run().info.run_id, nested=True):
        #         importance_path = "feature_importances.csv"
        #         self.feature_importances_.to_csv(importance_path, index=False)
        #         mlflow.log_artifact(importance_path)

    def _plot_results(self):
        """Generates plots for model comparison and bias-variance analysis."""
        if self.results_summary_.empty:
            print("No results to plot.")
            return

        # Plot 1: Model Comparison
        melted_results = self.results_summary_.melt(id_vars='model', var_name='metric', value_name='score')
        metrics_to_plot = [m for m in melted_results['metric'].unique() if m != 'best_params']
        melted_results = melted_results[melted_results['metric'].isin(metrics_to_plot)]

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(data=melted_results, x='score', y='model', hue='metric', orient='h', ax=ax)
        ax.set_title('Model Performance Comparison on Test Set', fontsize=16)
        ax.set_xlabel('Score')
        ax.set_ylabel('Model')
        ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        plt.show()

        # Plot 2: Learning Curve for the best model
        if self.verbose: print(f"\nGenerating learning curve for {self.best_model_name_}...")
        
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        train_sizes, train_scores, val_scores = learning_curve(
            estimator=self.best_model_,
            X=self.X,
            y=self.y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=cv,
            scoring=self.scoring,
            n_jobs=-1,
            random_state=self.random_state
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)

        fig_lc, ax_lc = plt.subplots(figsize=(10, 6))
        ax_lc.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        ax_lc.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")
        ax_lc.set_title(f'Learning Curve for {self.best_model_name_}', fontsize=16)
        ax_lc.set_xlabel("Training examples")
        ax_lc.set_ylabel("Score")
        ax_lc.legend(loc="best")
        ax_lc.grid()
        plt.show()

        # --- MLflow: Log learning curve plot for the best model ---
        # This part is commented out as it can cause issues with nested runs.
        # if self.use_mlflow and mlflow:
        #     with mlflow.start_run(run_id=mlflow.last_active_run().info.run_id, nested=True):
        #         lc_path = "learning_curve.png"
        #         fig_lc.savefig(lc_path)
        #         mlflow.log_artifact(lc_path)

        # Bias/Variance Diagnosis
        gap = train_scores_mean[-1] - val_scores_mean[-1]
        if val_scores_mean[-1] < 0.7 and gap < 0.1:
            print("Diagnosis: Potential High Bias (Underfitting). Both scores are low.")
        elif gap > 0.2:
            print(f"Diagnosis: Potential High Variance (Overfitting). Gap between train and validation score is {gap:.2f}.")
        else:
            print("Diagnosis: Model seems to have a reasonable bias-variance trade-off.")

    def run(self):
        """Executes the full training, tuning, and evaluation pipeline."""
        if self.use_mlflow and mlflow:
            mlflow.set_experiment(self.mlflow_experiment_name)
        elif self.use_mlflow:
            if self.verbose: print("MLflow not found. Skipping logging.")

        self._define_candidate_models()
        self._train_and_tune()
        self._evaluate_and_select_best()
        self._plot_results()

        if self.verbose:
            print("\n--- Final Results ---")
            print(self.results_summary_)
        
        if not self.use_mlflow:
            self.save_best_model(self.local_model_path)

        return self.best_model_, self.feature_importances_, self.features_used_

    def save_best_model(self, path: str, format: str = 'joblib'):
        """
        Saves the best trained model to a file.

        Args:
            path (str): The file path to save the model to.
            format (str): The format to save the model in. Currently only 'joblib' is supported.
        """
        if not self.best_model_:
            raise RuntimeError("No best model available to save. Run the trainer first.")

        if format == 'joblib':
            joblib.dump(self.best_model_, path)
            if self.verbose: print(f"Best model ({self.best_model_name_}) saved to {path}")
        else:
            raise ValueError(f"Unsupported save format: {format}. Use 'joblib'.")
