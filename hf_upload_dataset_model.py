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

# Repository names
DATASET_REPO_ID = "fahmizainal17/Smart-Waste-Dataset-Reduced"
MODEL_REPO_ID   = "fahmizainal17/Smart-Waste-Classifier-ResNet18"

# Paths to local assets
LOCAL_DATA_FOLDER = "data"
LOCAL_MODEL_FILE  = "model.pth"

# Initialize API
api = HfApi()

# Function to upload dataset
def push_dataset():
    print("==> Creating or reusing dataset repo:", DATASET_REPO_ID)
    api.create_repo(DATASET_REPO_ID, repo_type="dataset", token=HF_TOKEN, exist_ok=True)

    repo_slug = DATASET_REPO_ID.split("/")[-1]
    if os.path.isdir(repo_slug):
        shutil.rmtree(repo_slug)
    subprocess.run(["git", "clone", f"https://huggingface.co/datasets/{DATASET_REPO_ID}"], check=True)

    repo_folder = os.path.join(os.getcwd(), repo_slug)
    shutil.copytree(LOCAL_DATA_FOLDER, os.path.join(repo_folder, "data"), dirs_exist_ok=True)

    os.chdir(repo_folder)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Upload dataset"], check=True)
    subprocess.run(["git", "push"], check=True)
    os.chdir("..")
    print(f"==> Dataset uploaded to https://huggingface.co/datasets/{DATASET_REPO_ID}\n")

# Function to upload model
def push_model():
    print("==> Creating or reusing model repo:", MODEL_REPO_ID)
    api.create_repo(MODEL_REPO_ID, repo_type="model", token=HF_TOKEN, exist_ok=True)

    repo_slug = MODEL_REPO_ID.split("/")[-1]
    if os.path.isdir(repo_slug):
        shutil.rmtree(repo_slug)
    subprocess.run(["git", "clone", f"https://huggingface.co/{MODEL_REPO_ID}"], check=True)

    repo_folder = os.path.join(os.getcwd(), repo_slug)
    shutil.copy2(LOCAL_MODEL_FILE, os.path.join(repo_folder, "model.pth"))

    os.chdir(repo_folder)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Upload model"], check=True)
    subprocess.run(["git", "push"], check=True)
    os.chdir("..")
    print(f"==> Model uploaded to https://huggingface.co/{MODEL_REPO_ID}\n")

# Run the functions
if __name__ == "__main__":
    if os.path.isdir(LOCAL_DATA_FOLDER):
        push_dataset()
    else:
        print(f"WARNING: Data folder '{LOCAL_DATA_FOLDER}' not found. Skipping dataset upload.\n")

    if os.path.isfile(LOCAL_MODEL_FILE):
        push_model()
    else:
        print(f"WARNING: Model file '{LOCAL_MODEL_FILE}' not found. Skipping model upload.\n")
