# How to Use This

```bash
mlflow run . -P input_artifact="exercise_5/preprocessed_data.csv:latest" \
             -P artifact_root="data" \
             -P test_size=0.3 \
             -P stratify="genre"
```