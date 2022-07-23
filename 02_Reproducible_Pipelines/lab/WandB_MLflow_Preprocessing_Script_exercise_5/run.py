#!/usr/bin/env python
import argparse
import logging
import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    run = wandb.init(project="exercise_5", job_type="process_data")

    # Open the artifact: the name is not the filename,
    # but the name we used when registering it
    # To download the file we need to call .file()
    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    df = pd.read_parquet(artifact.file())
    
    logger.info("Performing pre-processing: duplicates + new data")
    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    # New feature
    # This feature will have to go to the feature store.
    # If you do not have a feature store,
    # then you should not compute it here as part of the preprocessing step.
    # Instead, you should compute it within the inference pipeline.
    df['title'].fillna(value='', inplace=True)
    df['song_name'].fillna(value='', inplace=True)
    df['text_feature'] = df['title'] + ' ' + df['song_name']

    # Persist cleaned dataset
    df.to_csv(args.artifact_name, sep=',', header=True, index=False)

    logger.info("Creating artifact")
    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    artifact.add_file(args.artifact_name)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    # Run finish is not necessary here
    # because it finishes automatically
    # when the script ends.
    # In a Jupyter session it is necessary, though.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)
