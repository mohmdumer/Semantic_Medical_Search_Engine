# 🩺 Semantic Medical Search Engine

## 🚨 Problem Statement

Doctors and patients often struggle to find relevant answers across vast medical forums or databases. Traditional keyword-based search engines lack the ability to understand the **semantic meaning** of medical queries, resulting in irrelevant or outdated responses.  
This project addresses this issue by building a **semantic search engine** specifically tailored for the medical domain using the **MedQuAD** dataset.

---

## 🎯 Project Objective

To develop a scalable, efficient semantic search engine that:

- Uses **medical sentence embeddings** to understand query intent
- Retrieves **top-N semantically relevant medical Q&A pairs**
- Exposes a simple **FastAPI** endpoint for integration and querying

---

## 🧠 Techniques and Tools Used

### 📦 Dataset

- **MedQuAD** (Medical Question Answering Dataset – CSV from Kaggle)

### 📚 Preprocessing

- Text normalization: punctuation removal, whitespace cleanup
- CSV parsing and column standardization

### 🤖 Embedding Model

- **Model**: `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`
- Built using **SentenceTransformers**
- Biomedical sentence-level embeddings with output normalization for cosine similarity

### ⚡ Vector Database

- **FAISS** (Facebook AI Similarity Search):
  - Index type: `IndexFlatIP`
  - Optimized for **cosine similarity** with normalized vectors

### 🔍 Semantic Search Logic

- Encode user query using the same embedding model
- Compute cosine similarity between query and medical questions
- Return **top-N** semantically similar questions and their expert answers

---

## 🖥️ API Interface

- Built with **FastAPI**
- Features:
  - `/search` endpoint to submit queries
  - Returns JSON with top-N results
  - Swagger UI and ReDoc enabled for easy testing

---

## 💻 Development Environment

- Python 3.10+
- VS Code (for development)
- Kaggle (for GPU-based embedding generation)

---

## 📁 Directory Structure

