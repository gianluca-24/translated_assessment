# pip install fastapi uvicorn
# uvicorn server:app --reload --port 8001

from fastapi import FastAPI, Query, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import re

# ----------------------------
# Initialize App and DB
# ----------------------------
app = FastAPI(title="Translation RAG Server")

# Initialize ChromaDB client and collection
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="translation_pairs"
)

# Load multilingual embedding model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ----------------------------
# API Endpoints
# ----------------------------

@app.post("/pairs")
async def add_translation_pair(request: Request):
    """
    Add a translation pair to the Chroma vector database.
    Returns {"message": "Ok"} if successful, or {"error": "..."} on failure.
    """
    try:
        data = await request.json()

        # Validate basic fields
        required_fields = ["id", "source_language", "target_language", "sentence", "translation"]
        for field in required_fields:
            if field not in data:
                return {"error": f"Missing required field: {field}"}

        # Create embedding
        embedding = model.encode(data["sentence"]).tolist()

        # Add or update entry in Chroma
        collection.add(
            ids=[data["id"]],
            embeddings=[embedding],
            documents=[data["sentence"]],
            metadatas=[{
                "source_language": data["source_language"],
                "target_language": data["target_language"],
                "sentence": data["sentence"],
                "translation": data["translation"]
            }]
        )

        return {"message": "Ok"}

    except Exception as e:
        # Catch all unexpected errors
        print(f"❌ Error adding translation pair: {e}")
        return False

@app.get("/prompt")
def get_translation_prompt(
    source_language: str = Query(..., description="Source language code"),
    target_language: str = Query(..., description="Target language code"),
    query_sentence: str = Query(..., description="Sentence to translate")
):
    """
    Retrieve up to 4 translation pairs similar to a query sentence using query parameters.
    Example URL:
    /prompt?source_language=en&target_language=it&query_sentence=Good+night
    """
    try:
        all_docs = collection.get()
        print(all_docs)
        # Encode the query sentence
        query_embedding = model.encode([query_sentence]).tolist()

        # Query ChromaDB with a language filter
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=4,
            where={
                "$and": [
                    {"source_language": source_language},
                    {"target_language": target_language}
                ]
            }
        )

        print(results)

        if not results or len(results["metadatas"][0]) == 0:
            # no similar found
            return {"prompt": False, "results": []}

        # Example variables
        srclang = source_language
        tgtlang = target_language
        query_text = query_sentence  # sentence to translate

        # Assume results["documents"][0] and results["metadatas"][0] contain the closest examples
        examples = list(zip(results["documents"][0], results["metadatas"][0]))

        # prompt construction from the literature: https://arxiv.org/pdf/2501.01679, titled 'Adaptive Few-shot Prompting for Machine Translation with Pre-trained Language Models
        prompt_lines = []
        prompt_lines.append(
            f"You are a professional translator. I will give you one or more examples of text fragments, where the first one is in {srclang} and the second one is the translation of the first fragment into {tgtlang}. These sentences will be displayed below."
        )

        # Add each example pair
        for i, (src, meta) in enumerate(examples, 1):
            prompt_lines.append(f"Example {i}. {srclang} text: {src}. {tgtlang} translation: {meta['translation']}")

        # Add the sentence to translate
        prompt_lines.append(
            f"\nAfter the example pairs, I will provide a {srclang} sentence and I would like you to translate it into {tgtlang}. "
            f"Please provide only the translation result without any additional comments, formatting, or chat content. Translate the text from {srclang} to {tgtlang}.\n"
        )
        prompt_lines.append(f"{srclang} sentence to translate: {query_text}")

        prompt = "\n".join(prompt_lines)

        return {"prompt": prompt, "results": results["metadatas"][0]}

    except Exception as e:
        print(f"❌ Error in /prompt: {e}")
        return {"prompt": False, "results": []}


@app.get("/stammering")
def detect_stammering(sentence: str = Query(..., description="Sentence to check for stammering")):
    """
    Detects if a sentence has stammering (repeated starting letters or syllables).
    Example: "I-I-I am fine", "B-b-bad weather"
    """
    # Simple regex-based detection of repeated characters
    has_stammer = bool(re.search(r'\b(\w)\1{1,}\w*', sentence.lower()))
    return {"has_stammer": has_stammer}

# ----------------------------
# Run: uvicorn server:app --reload --port 8000
# ----------------------------