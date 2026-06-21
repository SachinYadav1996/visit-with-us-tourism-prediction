import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_DATASET_REPO = "SachinY1996/visit-with-us-tourism-prediction"
HF_MODEL_REPO = "SachinY1996/visit-with-us-tourism-model"
MODEL_LOCAL_PATH = "tourism_project/model_building/best_tourism_model.joblib"

api = HfApi()

print("Downloading dataset from Hugging Face...")
X_train = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/Xtrain.csv")
X_test = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/Xtest.csv")
y_train = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/ytrain.csv").squeeze("columns")
y_test = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/ytest.csv").squeeze("columns")

numeric_features = X_train.select_dtypes(include="number").columns.tolist()
categorical_features = X_train.select_dtypes(include="object").columns.tolist()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))
])

param_grid = {
    "classifier__n_estimators": [50, 100],
    "classifier__max_depth": [5, 10]
}

print("Starting training and hyperparameter tuning...")
grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='f1', verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"F1-Score: {f1_score(y_test, y_pred)}")

os.makedirs("tourism_project/model_building", exist_ok=True)
joblib.dump(best_model, MODEL_LOCAL_PATH)

try:
    api.repo_info(repo_id=HF_MODEL_REPO, repo_type="model")
    print(f"Model repo '{HF_MODEL_REPO}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=HF_MODEL_REPO, repo_type="model", private=False)
    print(f"Created model repo '{HF_MODEL_REPO}'.")

api.upload_file(
    path_or_fileobj=MODEL_LOCAL_PATH,
    path_in_repo="best_tourism_model.joblib",
    repo_id=HF_MODEL_REPO,
    repo_type="model",
    commit_message="Upload best trained XGBoost model"
)

print("Model successfully trained and pushed to Hugging Face Model Hub!")
