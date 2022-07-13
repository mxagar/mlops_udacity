'''This script uses an artifact registered in Weights & Biases using arguments passed from the CLI.

To run it:

python use_artifact.py --artifact_name exercise_1/zen_of_python:v1

We can change between artifact versions with v0, v1, etc.

The different artifact versions are on the cloud and also locally, in the folder artifacts/
'''
import argparse
import logging
import pathlib
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    logger.info("Creating run in project exercise_1")
    run = wandb.init(project="exercise_1", job_type="use_file")

    logger.info("Getting artifact")

    # Get the artifact and store its local path in the variable "artifact_path"
    # From the W&B web interface (Usage):
    # artifact = run.use_artifact('datamix-ai/exercise_1/zen_of_python:v1', type='text_file')
    # Apparently, we don't necessarily need the team name datamix-ai
    artifact = run.use_artifact(args.artifact_name)
    # The file path is retreived, pointing to the correct version in folder artifacts/
    artifact_filepath = artifact.file()
    print(f"artifact_filepath = {artifact_filepath}")
    
    # We open the file
    # But 
    logger.info("Artifact content:")
    with open(artifact_filepath, "r") as fp:
        content = fp.read()

    print(content)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Use an artifact from W&B", fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name and version of W&B artifact", required=True
    )

    args = parser.parse_args()

    go(args)
