import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

SPACE_REPO = "SachinY1996/visit-with-us-tourism-app"
SPACE_SDK = "docker"
LOCAL_DEPLOYMENT_DIR = "tourism_project/deployment"

api = HfApi()

try:
    api.repo_info(repo_id=SPACE_REPO, repo_type="space")
    print(f"Space '{SPACE_REPO}' already exists.")
except RepositoryNotFoundError:
    create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        private=False,
        space_sdk=SPACE_SDK
    )
    print(f"Created Space '{SPACE_REPO}'.")

api.upload_folder(
    folder_path=LOCAL_DEPLOYMENT_DIR,
    repo_id=SPACE_REPO,
    repo_type="space",
    commit_message="Deploy Streamlit app to Hugging Face Space"
)

print(f"Deployment files uploaded successfully to Space: {SPACE_REPO}")
