import json

import mlflow
import os
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

    # Serialize random forest configuration
    model_config = os.path.abspath("random_forest_config.json")

    with open(model_config, "w+") as fp:
        json.dump(dict(config["random_forest"]), fp)

    _ = mlflow.run(
        os.path.join(root_path, "random_forest"),
        "main",
        parameters={
            "train_data": config["data"]["train_data"],
            "model_config": model_config
        },
    )


if __name__ == "__main__":
    go()
