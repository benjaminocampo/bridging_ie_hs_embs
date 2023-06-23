import mlflow
import pandas as pd
import glob
import os
from datasets import load_dataset
from transformers import HfApi, HfFolder

# Set the Hugging Face API token
HUGGINGFACE_TOKEN = "YOUR_HUGGINGFACE_TOKEN"
api = HfApi(token=HUGGINGFACE_TOKEN)

# Get all the MLflow runs and their metrics as a pandas DataFrame
runs = mlflow.search_runs(search_all_experiments=True)
runs_df = runs.loc[
    (runs["status"] == "FINISHED")
    & runs["tags.mlflow.runName"].apply(lambda t: "evaluated-in" not in t), [
        "tags.mlflow.runName", "tags.mlflow.note.content",
        "params.train.model.params.random_state", "artifact_uri"
    ]].sort_values(by="tags.mlflow.runName", ascending=False)

# Function to upload the model folder to the Hugging Face model hub
def upload_folder_to_hub(model_folder_path, repo_name, description):
    hf_folder = HfFolder(token=HUGGINGFACE_TOKEN)
    hf_folder.create_repo(repo_name, repo_type="model", exist_ok=True)
    hf_folder.push_to_hub(repo_name, folder=model_folder_path, commit_message="Initial commit", exist_ok=True)

    # Optionally, update the repository's metadata with the description
    api.update_repo_card(repo_name, {"description": description})

# Iterate over the runs and upload the model folders to the Hugging Face model hub
for index, run in runs_df.iterrows():
    run_name = run['tags.mlflow.runName']
    description = run['tags.mlflow.note.content']
    seed = run['params.train.model.params.random_state']
    artifact_uri = run["artifact_uri"][7:]
    paths = glob.glob(artifact_uri + "/**/model")
    model_path = paths[0]

    repo_name = f"YOUR_USER/{run_name}__seed-{seed}"

    # Upload the model folder to the Hugging Face model hub
    upload_folder_to_hub(model_path, repo_name, description)

# Confirm the upload by listing the models in the Hugging Face model hub
models = api.model_list()
for model in models:
    print(model.modelId)
