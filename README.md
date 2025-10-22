# Translation RAG Server

This project implements a **Retrieval-Augmented Generation (RAG) translation server** using **FastAPI**, **ChromaDB**, and **SentenceTransformers**. It allows you to store translation pairs in a vector database and retrieve the most similar examples for generating translations. I chose **ChromaDB** as the vector database for this project because it is simple to set up, lightweight, and extremely fast, making it ideal for storing and querying translation pairs efficiently. For measuring similarity between embeddings, I selected **cosine similarity** as the metric due to its effectiveness in capturing semantic similarity in high-dimensional spaces, which helps the RAG system retrieve the most relevant translation examples.


## Features

- Add translation pairs to a ChromaDB vector database.
- Retrieve up to 4 most similar translations for a given query.
- Detect stammering in sentences using a simple regex-based method.
- Supports multilingual embeddings with `SentenceTransformer`.

## Setup (Docker)

Build the Image:
```
docker build -t translation-rag-server .
```
Run the container:
```
docker run -d -p 8000:8000 --name rag-server translation-rag-server
```

## Setup (No Docker)

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
pip install -r requirements.txt
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
