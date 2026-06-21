import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Using your actual Hugging Face username instead of Nikidsouza23
REPO_ID = "SachinY1996/visit-with-us-tourism-prediction"
REPO_TYPE = "dataset"
LOCAL_DATA_DIR = "tourism_project/data"

# Auto-detects the logged-in token
api = HfApi()

try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Dataset repo '{REPO_ID}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False)
    print(f"Created dataset repo '{REPO_ID}'.")

api.upload_folder(
    folder_path=LOCAL_DATA_DIR,
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    commit_message="Upload raw tourism dataset"
)

print("Raw dataset uploaded successfully to Hugging Face!")
