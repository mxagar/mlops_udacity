#!/usr/bin/env python
import argparse
import logging
import pandas as pd
import wandb
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(project="exercise_13", job_type="test")

    logger.info("Downloading and reading test artifact")
    ## Get the args.test_data artifact from W&B locally
    ## YOUR CODE HERE
    test_data_artifact = run.use_artifact(args.test_data)
    test_data_path = test_data_artifact.file()
    df = pd.read_csv(test_data_path, low_memory=False)

    # Extract the target from the features
    logger.info("Extracting target from dataframe")
    X_test = df.copy()
    y_test = X_test.pop("genre")

    logger.info("Downloading and reading the exported model")

    ## Get the args.model_export artifact from W&B locally. Since this artifact contains a directory
    # and not a single file, you will have to use .download() instead of .file()
    ## YOUR CODE HERE
    # args.model_export
    artifact = run.use_artifact('datamix-ai/exercise_12/inference_pipeline:prod', type='pipeline')
    model_export_path = artifact.download()

    # Load the model using mlflow.sklearn.load_model
    ## YOUR CODE HERE
    pipe = mlflow.sklearn.load_model(model_export_path)

    # Compute the prediction from the model using .predict_proba on the test set
    ## YOUR CODE HERE
    # args.test_data    
    pred_proba = pipe.predict_proba(X_test)

    logger.info("Scoring")
    score = roc_auc_score(y_test, pred_proba, average="macro", multi_class="ovo")

    run.summary["AUC"] = score

    logger.info("Computing confusion matrix")
    fig_cm, sub_cm = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(
        pipe,
        X_test,
        y_test,
        ax=sub_cm,
        normalize="true",
        values_format=".1f",
        xticks_rotation=90,
    )
    fig_cm.tight_layout()

    run.log(
        {
            "confusion_matrix": wandb.Image(fig_cm)
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the provided model on the test artifact",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--model_export",
        type=str,
        help="Fully-qualified artifact name for the exported model to evaluate",
        required=True,
    )

    parser.add_argument(
        "--test_data",
        type=str,
        help="Fully-qualified artifact name for the test data",
        required=True,
    )

    args = parser.parse_args()

    go(args)
