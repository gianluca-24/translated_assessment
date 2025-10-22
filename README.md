# Translation RAG Server

This project implements a **Retrieval-Augmented Generation (RAG) translation server** using **FastAPI**, **ChromaDB**, and **SentenceTransformers**. It allows you to store translation pairs in a vector database and retrieve the most similar examples for generating translations.

---

## Features

- Add translation pairs to a ChromaDB vector database.
- Retrieve up to 4 most similar translations for a given query.
- Detect stammering in sentences using a simple regex-based method.
- Supports multilingual embeddings with `SentenceTransformer`.

---

## Setup

Clone the repository and navigate to the project folder:

```
git clone <repository_url>
cd <repository_folder>
```

Create and activate venv

mac: 
```
source rag_env/bin/activate
```
windows: 
```
.\rag_env\Scripts\Activate.ps1
```

Install dependencies

```
pip install fastapi uvicorn chromadb sentence-transformers
```

Run the server for development:
```
uvicorn server:app --reload --port 8001
```

Run the server for testing:
```
uvicorn server:app --port 8001
```

The prompt format was taken from: https://arxiv.org/pdf/2501.01679
