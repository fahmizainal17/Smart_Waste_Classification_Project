#!/usr/bin/env python

import os
import shutil
import subprocess
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load environment variables from .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("No HF_TOKEN found in .env file. Please add it.")

# Repository name for Hugging Face Space
SPACE_ID = "fahmizainal17/Smart-Waste-Classifier-App"

# Paths to local files for the app
LOCAL_APP_FILES = ["app.py", "requirements.txt", "model.pth"]

# Initialize API
api = HfApi()

# Function to deploy Gradio app to Hugging Face Spaces
def push_space():
    """Push Gradio app to an existing Hugging Face Space via Git."""
    print("==> Creating or reusing Space:", SPACE_ID)
    
    # Fix: Add space_sdk="gradio"
    api.create_repo(SPACE_ID, repo_type="space", space_sdk="gradio", token=HF_TOKEN, exist_ok=True)

    repo_slug = SPACE_ID.split("/")[-1]
    if os.path.isdir(repo_slug):
        shutil.rmtree(repo_slug)
    subprocess.run(["git", "clone", f"https://huggingface.co/spaces/{SPACE_ID}"], check=True)

    repo_folder = os.path.join(os.getcwd(), repo_slug)
    for file_name in LOCAL_APP_FILES:
        shutil.copy2(file_name, repo_folder)

    os.chdir(repo_folder)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Deploy Gradio app"], check=True)
    subprocess.run(["git", "push"], check=True)
    os.chdir("..")

    print(f"==> Space is live at https://huggingface.co/spaces/{SPACE_ID}\n")

# Run the function
if __name__ == "__main__":
    if all(os.path.isfile(f) for f in LOCAL_APP_FILES):
        push_space()
    else:
        print("WARNING: Some app file(s) not found. Skipping Space deployment.\n")
