# How to Use This

Since the MLflow project has default arguments, we can just run

```bash
mlflow run .
```

If we didn't have default arguments, we'd need to run

```bash
mlflow run . -P input_artifact="exercise_4/genres_mod.parquet:latest" \
             -P artifact_name="preprocessed_data.csv" \
             -P artifact_type=clean_data \
             -P artifact_description="Cleaned dataset"
```