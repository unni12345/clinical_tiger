# src/retriever.py
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue
from transformers import AutoTokenizer, AutoModel
from langchain_core.tools import tool
import torch
from langchain_openai import ChatOpenAI

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "clinical_chunks"

client = QdrantClient(location=QDRANT_URL, api_key=QDRANT_API_KEY)

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

# Load ClinicalBERT model and tokenizer
MODEL_ID = os.getenv("MODEL_ID")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)
model.eval()

# --- SECTION MAPPING (SIMPLIFIED INTENT-BASED VIEW) --- #
SECTION_MAPPING = {
    "dosage_lookup": ["DOSAGE AND ADMINISTRATION", "PROPER USE OF THIS MEDICATION"],
    "side_effects": ["SIDE EFFECTS AND WHAT TO DO ABOUT THEM", "ADVERSE REACTIONS", "REPORTING SIDE EFFECTS"],
    "contraindications": ["CONTRAINDICATIONS", "WARNINGS AND PRECAUTIONS"],
    "interactions": ["DRUG INTERACTIONS", "INTERACTIONS WITH THIS MEDICATION"],
    "indication_lookup": ["INDICATIONS AND CLINICAL USE"],
    "pharmacology": ["ACTION AND CLINICAL PHARMACOLOGY", "PHARMACOLOGY"],
    "storage_handling": ["HOW TO STORE IT", "STORAGE AND STABILITY"],
    "composition": ["COMPOSITION", "PHARMACEUTICAL INFORMATION"],
    "overdose": ["OVERDOSAGE", "SYMPTOMS AND TREATMENT OF OVERDOSAGE"],
    "visual": ["TABLE", "IMAGE"],
    "general_info": ["ABOUT THIS MEDICATION", "MORE INFORMATION"]
}

SECTION_LIST = sorted(set(s for sl in SECTION_MAPPING.values() for s in sl))

def suggest_sections_from_query(query: str, top_n: int = 3) -> list[str]:
    """
    Suggest up to `top_n` document sections relevant to the query.
    
    Returns a list of valid section names.
    """
    prompt = (
        f"You are a clinical assistant. Based on the user's query, suggest up to {top_n} relevant document sections from the following list:\n"
        f"{', '.join(SECTION_LIST)}\n"
        f"\nQuery: {query}\n"
        f"Return only a list of section names."
    )
    response = llm.invoke(prompt)
    return [sec.strip() for sec in response.content.split("\n") if sec.strip() in SECTION_LIST]


def embed_query(query: str):
    """
    Embed the query using the ClinicalBERT model and return the embedding vector.
    """
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    return cls_vector


def retrieve_top_k(query: str, k: int = 5, filter_types: list[str] = ["text"], allowed_sources: list[str] = None):
    """
    Retrieve the top k points matching the query vector with specified type and source filters.
    """
    query_vector = embed_query(query)
    
    must_conditions = [
        FieldCondition(
            key="type",
            match=MatchAny(any=filter_types)
        ),
        FieldCondition(
            key="is_searchable",
            match=MatchValue(value=True)
        )
    ]
    
    # Add a condition to filter by allowed sources if provided.
    if allowed_sources:
        print("Allowed sources: ", allowed_sources)
        source_filters = [FieldCondition(key="source", match=MatchValue(value=allowed_sources))]
        must_conditions.extend(source_filters)
    
    search_filter = Filter(must=must_conditions)
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        with_payload=True,
        limit=k,
    ).points
    return results


def retrieve_top_k_sectional(
    query: str,
    k: int = 5,
    allowed_sections: list[str] = ["UNSECTIONED"],
    allowed_sources: list[str] = None
):
    """
    Retrieve top k points for specified sections and optionally sources.
    """
    query_vector = embed_query(query)
    print("Allowed sections: ", allowed_sections)
    must_filter = [FieldCondition(key="section", match=MatchValue(value=section)) for section in allowed_sections]
    
    # If allowed_sources is provided, add filter conditions for the source field.
    if allowed_sources:
        print("Allowed sources: ", allowed_sources)
        source_filters = [FieldCondition(key="source", match=MatchValue(value=allowed_sources))]
        must_filter.extend(source_filters)
    
    search_filter = Filter(must=must_filter)
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        with_payload=True,
        limit=k,
    ).points
    return results


# @tool
# def retrieve_context(query: str, top_k: int = 3) -> str:
#     """Retrieve top-k relevant text chunks based on the query."""
#     print("Insde retrieve_context")
#     results = retrieve_top_k(query, k=top_k, filter_types=["text"])
#     return "\n\n".join([res.payload["text"] for res in results])

# @tool
# def retrieve_table(query: str, top_k: int = 2) -> str:
#     """Retrieve top-k relevant table chunks based on the query."""
#     print("Inside retrieve_table")
#     results = retrieve_top_k(query, k=top_k, filter_types=["table"])
#     return "\n---\n".join([res.payload["text"] for res in results])

# @tool
# def retrieve_image_caption(query: str, top_k: int = 1) -> str:
#     """Retrieve top-k image captions based on the query."""
#     print("Inside retrieve_image_caption")
#     results = retrieve_top_k(query, k=top_k, filter_types=["image"])
#     return "\n\n".join([res.payload["text"] + f"\n(Image: {res.payload.get('image_path')})" for res in results])

# @tool
# def retrieve_image_path(query: str, top_k: int = 1) -> str:
#     """Retrieve top-k image path based on the query."""
#     print("Inside retrieve_image_path")
#     results = retrieve_top_k(query, k=top_k, filter_types=["image"])
#     return results.get('image_path')

if __name__ == "__main__":
    hits = retrieve_top_k("What is the dosage for elderly patients taking Lipitor?", k=5)
    for hit in hits:
        print("Score:", hit.score)
        print("Text:", hit.payload.get("text"))
        print("Source:", hit.payload.get("source"))
        print("Type:", hit.payload.get("type"))
        print("Page:", hit.payload.get("page"))
        print("---")
