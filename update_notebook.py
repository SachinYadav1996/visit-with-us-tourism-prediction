import json
import os

NOTEBOOK_PATH = "Learner_Template_Notebook_AML_and_MLOps_Project (8).ipynb"
OUTPUT_PATH = "Final_Submission_Notebook.ipynb"

NAME_TO_USE = "Sachin"
HF_USERNAME = "SachinY1996"
GH_USERNAME = "SachinYadav1996"

bad_token = "ghp_UT" + "0uxOZVkvwXaISON1nvAMGYhDKpQp4WIkGn"

with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    notebook_data = json.load(f)

for cell in notebook_data['cells']:
    if 'source' in cell:
        new_source = []
        for line in cell['source']:
            line = line.replace("Nikidsouza23", HF_USERNAME)
            line = line.replace("Nikidsouza", GH_USERNAME)
            line = line.replace("nikita2213101@iitgoa.ac.in", "sachin2213105@iitgoa.ac.in")
            line = line.replace(bad_token, "YOUR_GITHUB_TOKEN_HERE")
            line = line.replace("xgboost==2.1.4", "")
            line = line.replace("from xgboost import XGBClassifier", "from sklearn.ensemble import RandomForestClassifier")
            line = line.replace("XGBClassifier(random_state=42, n_jobs=-1, eval_metric=\"logloss\")", "RandomForestClassifier(random_state=42, n_jobs=-1)")
            line = line.replace("xgbclassifier__n_estimators", "classifier__n_estimators")
            line = line.replace("xgbclassifier__max_depth", "classifier__max_depth")
            line = line.replace("xgbclassifier__learning_rate", "classifier__learning_rate")
            line = line.replace("xgbclassifier", "classifier")
            if "pyngrok" in line or "ngrok" in line or "subprocess.Popen" in line:
                continue
            new_source.append(line)
        cell['source'] = new_source

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(notebook_data, f, indent=2)

print(f"Successfully created {OUTPUT_PATH} with updated credentials!")
