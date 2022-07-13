import mlflow
import os
import wandb
import hydra
from omegaconf import DictConfig


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    print(root_path)
    
    # python download_data.py --file_url {file_url} \
    #                     --artifact_name {artifact_name} \
    #                     --artifact_type {artifact_type} \
    #                     --artifact_description {artifact_description}
    #
    # python download_data.py \
    #    --file_url https://raw.githubusercontent.com/scikit-learn/scikit-learn/4dfdfb4e1bb3719628753a4ece995a1b2fa5312a/sklearn/datasets/data/iris.csv \
    #    --artifact_name iris \
    #    --artifact_type raw_data \
    #    --artifact_description "The sklearn IRIS dataset"
    #
    # mlflow run . \
    # -P file_url=https://raw.githubusercontent.com/scikit-learn/scikit-learn/4dfdfb4e1bb3719628753a4ece995a1b2fa5312a/sklearn/datasets/data/iris.csv \
    # -P artifact_name=iris \
    # -P artifact_description="The sklearn IRIS dataset"
    
    _ = mlflow.run(
        os.path.join(root_path, "download_data"),
        "main",
        parameters={
            "file_url": config["data"]["file_url"],
            "artifact_name": "iris.csv",
            "artifact_type": "raw_data",
            "artifact_description": "Input data"
        }
    )

    # python run.py --input_artifact {input_artifact} \
    #           --artifact_name {artifact_name} \
    #           --artifact_type {artifact_type} \
    #           --artifact_description {artifact_description}
    #
    # mlflow run . \
    # -P input_artifact=iris.csv:latest \
    # -P artifact_name=clean_data.csv \
    # -P artifact_type=processed_data \
    # -P artifact_description="Cleaned data"
    _ = mlflow.run(
        os.path.join(root_path, "process_data"),
        "main",
        parameters={
            "input_artifact": "iris.csv:latest",
            "artifact_name": "clean_data.csv",
            "artifact_type": "processed_data",
            "artifact_description": "Cleaned data"
        }
    )



if __name__ == "__main__":
    go()
