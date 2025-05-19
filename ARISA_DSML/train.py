from pathlib import Path
import mlflow.sklearn
import optuna
import mlflow
from mlflow.client import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from loguru import logger
import nannyml as nml
import joblib
import plotly.graph_objects as go

from ARISA_DSML.helpers import get_git_commit_hash
from ARISA_DSML.config import (
    FIGURES_DIR,
    MODEL_NAME,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    features,
    target,
)


def run_hyperopt(X_train:pd.DataFrame, y_train:pd.DataFrame, test_size:float=0.25, n_trials:int=20, overwrite:bool=False)->str|Path:  # noqa: PLR0913
    """Run optuna hyperparameter tuning."""

    best_params_path = MODELS_DIR / "best_params.pkl"
    if not best_params_path.is_file() or overwrite:
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(X_train, y_train, test_size=test_size, random_state=42)

        def objective(trial:optuna.trial.Trial)->float:
            with mlflow.start_run(nested=True):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 2, 10, log=True),
                    "max_depth": trial.suggest_int("max_depth", 2, 32),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                }

                model = RandomForestRegressor(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    min_samples_split=params["min_samples_split"],
                    min_samples_leaf=params["min_samples_leaf"],
                    random_state=42,
                )
                model.fit(X_train_opt, y_train_opt)

                mlflow.log_params(params)
                preds = model.predict(X_val_opt)

                mae = mean_absolute_error(y_val_opt, preds)
                mse = mean_squared_error(y_val_opt, preds)
                r2 = r2_score(y_val_opt, preds)

                mlflow.log_metric("mae", mae)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r2", r2)

            return mae
        # Create a study object
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        joblib.dump(study.best_params, best_params_path)

        params = study.best_params
    else:
        params = joblib.load(best_params_path)
    logger.info("Best Parameters: " + str(params))
    return best_params_path


def train_cv(X_train:pd.DataFrame, y_train:pd.DataFrame)->str|Path:
    """Do cross-validated training."""

    params = {
        'n_estimators': [2, 4, 10],
        'max_depth': [2, 10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    random_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=params,
        scoring='neg_mean_absolute_error',
        cv=5,
        verbose=1,
        n_jobs=-1,
        return_train_score=True,
        random_state=42,
    )
    random_search.fit(X_train, y_train)

    cv_results = pd.DataFrame(random_search.cv_results_)
    cv_results['iteration'] = range(0, len(cv_results))
    cv_output_path = MODELS_DIR / "cv_results.csv"
    cv_results.to_csv(cv_output_path, index=False)
    return cv_output_path


def train(X_train:pd.DataFrame, y_train:pd.DataFrame,
          params:dict|None, artifact_name:str="random_forest_regressor_model_wind_power", cv_results=None,
          )->tuple[str|Path]:
    """Train model on full dataset without cross-validation."""
    if params is None:
        logger.info("Training model without tuned hyperparameters")
        params = {}
    with mlflow.start_run():

        model = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 10),
            max_depth=params.get("max_depth", 10),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=42,
            verbose=True,
        )

        model.fit(
            X_train,
            y_train
        )
        params["feature_columns"] = features
        mlflow.log_params(params)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        model_path = MODELS_DIR / f"{artifact_name}.cbm"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        cv_metric_mean = cv_results["mean_test_score"].mean()
        mlflow.log_metric("score_cv_mean", cv_metric_mean)

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train,
            registered_model_name=MODEL_NAME
        )

        client = MlflowClient(mlflow.get_tracking_uri())
        model_info = client.get_latest_versions(MODEL_NAME)[0]
        client.set_registered_model_alias(MODEL_NAME, "challenger", model_info.version)
        client.set_model_version_tag(
            name=model_info.name,
            version=model_info.version,
            key="git_sha",
            value=get_git_commit_hash(),
        )
        model_params_path = MODELS_DIR / "model_params.pkl"
        joblib.dump(params, model_params_path)
        fig1 = plot_error_scatter(
            df_plot=cv_results,
            name="Mean MAE Score",
            title="Cross-Validation (N=5) Mean MAE score with Error Bands",
            xtitle="Training Steps",
            ytitle="Performance Score",
            yaxis_range=[0.5, 1.0],
        )
        mlflow.log_figure(fig1, "test-MAE-mean_vs_iterations.png")

        """----------NannyML----------"""
        # Model monitoring initialization
        reference_df = X_train.copy()
        reference_df["prediction"] = model.predict(X_train)
        reference_df[target] = y_train
        chunk_size = 50

        # univariate drift for features
        udc = nml.UnivariateDriftCalculator(
            column_names=X_train.columns,
            chunk_size=chunk_size,
        )
        udc.fit(reference_df.drop(columns=["prediction", target]))

        # Confidence-based Performance Estimation for target
        # For regression, use RegressionPerformanceCalculator instead of CBPE
        estimator = nml.PerformanceCalculator(
            metrics=["mae"],
            y_pred="prediction",
            problem_type="regression",
            y_true=target,
            chunk_size=chunk_size,
        )
        estimator = estimator.fit(reference_df)

        store = nml.io.store.FilesystemStore(root_path=str(MODELS_DIR))
        store.store(udc, filename="udc.pkl")
        store.store(estimator, filename="estimator.pkl")

        mlflow.log_artifact(MODELS_DIR / "udc.pkl")
        mlflow.log_artifact(MODELS_DIR / "estimator.pkl")

    return (model_path, model_params_path)


def plot_error_scatter(  # noqa: PLR0913
        df_plot:pd.DataFrame,
        x:str="iteration",
        y:str="mean_test_score",
        err:str="std_test_score",
        name:str="",
        title:str="",
        xtitle:str="",
        ytitle:str="",
        yaxis_range:list[float]|None=None,
    )->None:
    """Plot plotly scatter plots with error areas."""
    # Create figure
    fig = go.Figure()

    if not len(name):
        name = y

    # Add mean performance line
    fig.add_trace(
        go.Scatter(
            x=df_plot[x], y=df_plot[y], mode="lines", name=name, line={"color": "blue"},
        ),
    )

    # Add shaded error region
    fig.add_trace(
        go.Scatter(
            x=pd.concat([df_plot[y], df_plot[x][::-1]]),
            y=pd.concat([df_plot[y]+df_plot[err],
                         df_plot[y]-df_plot[err]]),
            fill="toself",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line={"color":"rgba(255, 255, 255, 0)"},
            showlegend=False,
        ),
    )

    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        template="plotly_white",
    )

    if yaxis_range is not None:
        fig.update_layout(
            yaxis={"range": yaxis_range},
        )

    fig.show()
    fig.write_image(FIGURES_DIR / f"{y}_vs_{x}.png")
    return fig


def get_or_create_experiment(experiment_name:str):
    """Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters
    ----------
    - experiment_name (str): Name of the MLflow experiment.

    Returns
    -------
    - str: ID of the existing or newly created MLflow experiment.

    """
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id

    return mlflow.create_experiment(experiment_name)


if __name__=="__main__":
    # for running in workflow in actions again again
    df_train = pd.read_csv(PROCESSED_DATA_DIR / "Location1.csv")

    y_train = df_train.pop(target)
    X_train = df_train[features]

    experiment_id = get_or_create_experiment("wind_power_hyperparam_tuning")
    mlflow.set_experiment(experiment_id=experiment_id)
    best_params_path = run_hyperopt(X_train, y_train)
    params = joblib.load(best_params_path)
    print(params)
    cv_output_path = train_cv(X_train, y_train)
    cv_results = pd.read_csv(cv_output_path)

    experiment_id = get_or_create_experiment("wind_power_full_training")
    mlflow.set_experiment(experiment_id=experiment_id)
    model_path, model_params_path = train(X_train, y_train, params, cv_results=cv_results)

    cv_results = pd.read_csv(cv_output_path)
