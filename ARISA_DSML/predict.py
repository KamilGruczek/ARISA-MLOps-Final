from pathlib import Path
import pandas as pd
from loguru import logger
import mlflow.sklearn
import mlflow
from mlflow.client import MlflowClient
import json
import shap
import os
import nannyml as nml
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from ARISA_DSML.config import FIGURES_DIR, MODELS_DIR, target, PROCESSED_DATA_DIR, MODEL_NAME
from ARISA_DSML.resolve import get_model_by_alias


def plot_shap(model:RandomForestRegressor, df_plot:pd.DataFrame)->None:
    """Plot model shapley overview plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_plot)

    shap.summary_plot(shap_values, df_plot, show=False)
    plt.savefig(FIGURES_DIR / "test_shap_overall.png")


def predict(model:RandomForestRegressor, df_pred:pd.DataFrame, params:dict, probs=False)->str|Path:
    """Do predictions on test data."""

    feature_columns = params.pop("feature_columns")

    preds = model.predict(df_pred[feature_columns])

    plot_shap(model, df_pred[feature_columns])
    df_pred[target] = preds
    preds_path = MODELS_DIR / "preds.csv"
    df_pred[["Time", target]].to_csv(preds_path, index=False)

    return preds_path


if __name__=="__main__":
    df_test = pd.read_csv(PROCESSED_DATA_DIR / "Location1.csv")

    client = MlflowClient(mlflow.get_tracking_uri())
    model_info = get_model_by_alias(client, alias="champion")
    if model_info is None:
        logger.info("No champion model, predicting using newest model")
        model_info = client.get_latest_versions(MODEL_NAME)[0]

    # extract params/metrics data for run `test_run_id` in a single dict
    run_data_dict = client.get_run(model_info.run_id).data.to_dictionary()
    run = client.get_run(model_info.run_id)
    log_model_meta = json.loads(json.dumps(run.data.params))
    log_model_meta

    _, artifact_folder = os.path.split(model_info.source)
    logger.info(artifact_folder)
    model_uri = "runs:/{}/{}".format(model_info.run_id, artifact_folder)
    logger.info(model_uri)
    loaded_model = mlflow.sklearn.load_model(model_uri)

    local_path = client.download_artifacts(model_info.run_id, "udc.pkl", "models")
    local_path = client.download_artifacts(model_info.run_id, "estimator.pkl", "models")

    store = nml.io.store.FilesystemStore(root_path=str(MODELS_DIR))
    udc = store.load(filename="udc.pkl", as_type=nml.UnivariateDriftCalculator)
    estimator = store.load(filename="estimator.pkl", as_type=nml.PerformanceCalculator)

    params = run_data_dict["params"]
    params["feature_columns"] = eval(log_model_meta['feature_columns'])
    preds_path = predict(loaded_model, df_test, params, probs=True)

    df_preds = pd.read_csv(preds_path)

    analysis_df = df_test.copy()
    analysis_df["prediction"] = df_preds[target]

    from ARISA_DSML.helpers import get_git_commit_hash
    git_hash = get_git_commit_hash()
    mlflow.set_experiment("wind_power_predictions")
    with mlflow.start_run(tags={"git_sha": get_git_commit_hash()}):
        estimated_performance = estimator.calculate(analysis_df)
        fig1 = estimated_performance.plot()
        mlflow.log_figure(fig1, "estimated_performance.png")
        univariate_drift = udc.calculate(analysis_df.drop(columns=["prediction"], axis=1))
        plot_col_names = analysis_df.drop(columns=["prediction"], axis=1).columns
        for p in plot_col_names:
            try:
                fig2 = univariate_drift.filter(column_names=[p]).plot()
                mlflow.log_figure(fig2, f"univariate_drift_{p}.png")
                fig3 = univariate_drift.filter(period="analysis", column_names=[p]).plot(kind='distribution')
                mlflow.log_figure(fig3, f"univariate_drift_dist_{p}.png")
            except Exception:
                logger.info("failed to plot some univariate drift analyses!")
        mlflow.log_params({"git_hash": git_hash})
