# src/embed_and_store.py
import os
import json
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct

# Load environment variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "clinical_chunks"

# Load ClinicalBERT
MODEL_ID = os.getenv("MODEL_ID")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)
model.eval()

# Initialize Qdrant client
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Create or reset the collection
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

# Load chunks
with open("data/chunks.json") as f:
    chunks = json.load(f)

# Embed function (CLS pooling)
def get_cls_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    return cls_vector

# Upload text chunks with vectors
vectors = []
payloads = []
ids = []

# Delete the collection if it exists, then create a new one.
if client.collection_exists(collection_name=COLLECTION_NAME):
    client.delete_collection(collection_name=COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

# Create collection
for idx, chunk in enumerate(chunks):
    embedding = get_cls_embedding(chunk["text"])
    vectors.append(embedding)
    payload = {
        "text": chunk["text"],
        "page": chunk["page"],
        "source": chunk["source"],
        "chunk_id": chunk["chunk_id"],
        "section": chunk.get("section"),
        "type": chunk["type"],
        "is_searchable": True,
        "image_path": chunk.get("image_path", ""),
        "url": chunk.get("url", "")
    }
    payloads.append(payload)
    ids.append(idx)
last_id = idx

# Upload the created collection
client.upload_collection(
    collection_name=COLLECTION_NAME,
    vectors=vectors,
    payload=payloads,
    ids=ids,
)
print(f"✅ Uploaded {len(vectors)} text chunks to Qdrant.")
