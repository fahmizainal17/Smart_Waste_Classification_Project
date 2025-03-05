# **📦 Smart Waste Classification**

## **1. Screenshots**

### **1.1 The App Interface**  
![The Apps](images/The_apps.png)  
The Gradio-based web application for waste classification.

### **1.2 Classification Result**  
![The Result](images/The_result.png)  
The classification output showing the predicted waste category.

---

## **2. Overview**
This project is an AI-powered **trash classification system** that identifies different types of waste: **cardboard, glass, metal, paper, plastic, and trash**. It includes:
- **Dataset preparation & upload** to Hugging Face.
- **Model training using ResNet18**.
- **Deployment of a Gradio app** on Hugging Face Spaces.

🔗 **Live Demo:** [Smart-Waste-Classifier-App](https://huggingface.co/spaces/fahmizainal17/Smart-Waste-Classifier-App)

---

## **3. Table of Contents**
1. [Key Features](#4-key-features)  
2. [Technology Stack](#5-technology-stack)  
3. [Project Structure](#6-project-structure)  
4. [Installing & Running](#7-installing--running)  
5. [Processing Pipeline](#8-processing-pipeline)  
6. [Data Analysis](#9-data-analysis)  
7. [Deployment Steps](#10-deployment-steps)  
8. [References](#11-references)  
9. [License](#12-license)  

---

## **4. Key Features**

- **Automated Waste Classification**  
  - Supports **cardboard, glass, metal, paper, plastic, trash**.
  - Uses **ResNet18** for classification.

- **Robust Data Processing**  
  - Dataset pre-processing and augmentation.
  - Normalization and resizing for ResNet.

- **Fast Deployment**  
  - Hosted on **Hugging Face Spaces**.
  - Runs efficiently in a lightweight environment.

---

## **5. Technology Stack**

- **Python 3.12+**  
- **PyTorch & Torchvision**  
- **Gradio** for UI  
- **Hugging Face Hub** for dataset & model storage  

---

## **6. Project Structure**

```plaintext
Smart_Waste_Classification_Project/
├── README.md
├── images/
│   ├── cardboard.jpg
│   ├── The_apps.png
│   ├── The_result.png
├── data/
│   ├── train/
│   ├── val/
├── model.pth
├── train.py
├── app.py
├── hf_upload_dataset_model.py
├── hf_deploy_space.py
├── requirements.txt
└── .env
```

---

## **7. Installing & Running**

### **Step 1: Install Dependencies**
```bash
pip install --upgrade huggingface_hub transformers requests tqdm torch torchvision gradio python-dotenv
```

### **Step 2: Set Up Hugging Face Token**
Create a `.env` file and add:
```bash
HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### **Step 3: Upload Dataset & Model**
```bash
python hf_upload_dataset_model.py
```
This uploads:
✅ **Dataset** → [Smart-Waste-Dataset-Reduced](https://huggingface.co/datasets/fahmizainal17/Smart-Waste-Dataset-Reduced)  
✅ **Model** → [Smart-Waste-Classifier-ResNet18](https://huggingface.co/fahmizainal17/Smart-Waste-Classifier-ResNet18)

---

## **8. Processing Pipeline**
```mermaid
graph TD
    A[Image Input] --> B[Data Preprocessing]
    B --> C[Feature Extraction]
    C --> D[ResNet18 Model]
    D --> E[Classification Output]
```

1. **Preprocess images** (resize, normalize).
2. **Extract features** using **ResNet18**.
3. **Classify into six categories**.

---

## **9. Data Analysis**

### **Sample Classification Result**
```plaintext
Prediction: Cardboard (75%)
Top 3 Classes:
1. Cardboard - 75%
2. Paper - 21%
3. Trash - 2%
```

---

## **10. Deployment Steps**

### **Deploy Gradio App on Hugging Face**
```bash
python hf_deploy_space.py
```
This deploys to **Hugging Face Spaces**:
✅ **Live App** → [Smart-Waste-Classifier-App](https://huggingface.co/spaces/fahmizainal17/Smart-Waste-Classifier-App)

---

## **11. References**
- [Hugging Face Docs](https://huggingface.co/docs/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Gradio Guide](https://gradio.app/docs/)

---

## **12. License**
Copyright (c) 2023 Fahmi Zainal

This project is licensed for personal and educational use only. Modification and redistribution require explicit permission from the author.

🚀 **Enjoy using Smart Waste Classifier!**

