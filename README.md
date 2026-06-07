# Agentic Medical AI Assistant for Chest X-Ray Analysis

## Overview

Agentic Medical AI Assistant is an end-to-end AI-powered healthcare application that analyzes chest X-ray images, predicts multiple thoracic diseases, retrieves relevant medical knowledge, and generates radiology-style reports using Large Language Models (LLMs).

The system combines Deep Learning, Retrieval-Augmented Generation (RAG), LangChain, LangGraph, and Explainable AI techniques to create an intelligent workflow capable of assisting medical image interpretation.

The project was developed using the NIH ChestX-ray14 dataset and follows a modular agent-based architecture.

---

## Key Features

### Multi-Label Disease Classification

A DenseNet121 Convolutional Neural Network is trained and fine-tuned to detect multiple chest diseases simultaneously:

* Atelectasis
* Cardiomegaly
* Effusion
* Infiltration
* Pneumothorax

The model outputs probability scores for each disease rather than a single diagnosis.

---

### Explainable AI with Grad-CAM

The system provides visual explanations for model predictions using Gradient-weighted Class Activation Mapping (Grad-CAM).

Grad-CAM highlights image regions that contributed most strongly to a disease prediction, improving transparency and interpretability.

---

### Retrieval-Augmented Generation (RAG)

Predicted diseases are used to retrieve relevant medical knowledge from a vector database.

The RAG pipeline consists of:

* Sentence Transformers embeddings
* FAISS Vector Store
* Medical knowledge documents

This retrieved information is supplied to the LLM as context during report generation.

---

### AI-Powered Report Generation

Using LangChain and Ollama, the system generates structured radiology-style reports.

Generated reports contain:

* Findings
* Impression
* Recommendations

The report generation process is enhanced using retrieved medical knowledge.

---

### Agent-Based Workflow with LangGraph

The application is implemented as a multi-agent workflow.

Agents include:

#### Prediction Agent

Performs chest X-ray disease classification.

#### Retrieval Agent

Retrieves disease-specific medical knowledge from FAISS.

#### Report Agent

Generates final radiology reports using Qwen 2.5.

The workflow is orchestrated using LangGraph.

---

### Interactive Web Application

A Streamlit-based user interface allows users to:

* Upload chest X-ray images
* View disease probabilities
* Generate AI-assisted reports
* Analyze model predictions interactively

---

## System Architecture

```text
Chest X-Ray Image
        │
        ▼
DenseNet121 CNN
        │
        ▼
Prediction Agent
        │
        ▼
Disease Probabilities
        │
        ▼
Retrieval Agent
        │
        ▼
FAISS Knowledge Retrieval
        │
        ▼
Retrieved Medical Context
        │
        ▼
Report Agent
        │
        ▼
Qwen 2.5 (LLM)
        │
        ▼
Radiology Report
```

---

## Technology Stack

### Deep Learning

* PyTorch
* Torchvision
* DenseNet121

### Data Processing

* NumPy
* Pandas
* Pillow
* OpenCV

### Explainable AI

* Grad-CAM

### Retrieval-Augmented Generation

* Sentence Transformers
* FAISS

### LLM & Agents

* LangChain
* LangGraph
* Ollama
* Qwen 2.5

### Frontend

* Streamlit

### Deployment

* Docker

---

## Dataset

### NIH ChestX-ray14 Dataset

The project uses the NIH ChestX-ray14 dataset containing over 100,000 chest X-ray images with 14 thoracic disease labels.

Dataset used:

* NIH ChestX-ray14
* Chest Radiography Images
* Frontal Chest X-rays

Selected disease classes:

* Atelectasis
* Cardiomegaly
* Effusion
* Infiltration
* Pneumothorax

---

## Project Structure

```text
medical_llm/

├── data/
│   ├── knowledge/
│   ├── train.csv
│   ├── val.csv
│   └── encoded_labels.csv
│
├── models/
│
├── src/
│   ├── cnn/
│   ├── rag/
│   ├── llm/
│   ├── agents/
│   ├── graph/
│   ├── tests/
│   └── utils/
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Agent Workflow

### Prediction Agent

Input:

```text
Chest X-Ray
```

Output:

```text
Disease probabilities
```

---

### Retrieval Agent

Input:

```text
Predicted diseases
```

Output:

```text
Relevant medical knowledge
```

---

### Report Agent

Input:

```text
Predictions + Retrieved Context
```

Output:

```text
Radiology Report
```

---

## Running the Application

### Clone Repository

```bash
git clone https://github.com/your-username/agentic-medical-ai-assistant.git

cd agentic-medical-ai-assistant
```

### Create Environment

```bash
python -m venv med_llm

source med_llm/bin/activate
```

Windows:

```bash
med_llm\Scripts\activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Start Ollama

```bash
ollama serve
```

Pull model:

```bash
ollama pull qwen2.5
```

---

### Launch Streamlit

```bash
streamlit run src/app.py
```

---

## Docker Deployment

Build image:

```bash
docker build -t medical-ai .
```

Run container:

```bash
docker run -p 8501:8501 medical-ai
```

Open:

```text
http://localhost:8501
```

---

## Results

The model successfully performs:

* Multi-label disease classification
* Knowledge retrieval
* AI-assisted report generation

Evaluation metrics include:

* Precision
* Recall
* F1 Score
* ROC-AUC

The system demonstrates strong performance for Cardiomegaly, Effusion, Atelectasis, Infiltration, and Pneumothorax detection.

---

## Future Improvements

* Additional disease categories
* Larger medical knowledge base
* Advanced RAG pipelines
* Multi-agent verification workflows
* Clinical decision support integration
* Cloud deployment
* Real-time hospital integration

---

## Disclaimer

This project is intended for educational and research purposes only.

The generated reports are AI-assisted outputs and should not be used for clinical diagnosis, treatment decisions, or medical advice. Always consult qualified healthcare professionals for medical interpretation.

---

## Author

Rishav Dasgupta

Computer Science Engineer | AI & Machine Learning Enthusiast | Full Stack Developer

Specializations:

* Deep Learning
* Generative AI
* Agentic AI Systems
* Retrieval-Augmented Generation (RAG)
* MLOps
