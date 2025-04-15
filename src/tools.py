# src/tools.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from retriever import (
    retrieve_top_k,
    retrieve_top_k_sectional,
    SECTION_MAPPING
)
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

# Initialize the base LLM with a clinical assistant role.
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    model_kwargs={
        "messages": [
            {
                "role": "system",
                "content": "You are a clinical assistant. Always answer truthfully."
            }
        ]
    }
)

@tool
def retrieve_image_path(query: str, top_k: int = 1) -> tuple[str, str]:
    """
    Retrieve the top image path and its caption based on the query.
    """
    print("Inside retrieve_image_path")
    results = retrieve_top_k(query, k=top_k, filter_types=["image"])
    print("Results: ", results)
    if results and len(results) > 0:
        image_path = results[0].payload.get("image_path", "")
        caption = results[0].payload.get("text", "")
        return image_path, caption
    return "", ""

@tool
def classify_sections(query: str) -> list:
    """
    Classify the user's query into one or more relevant section categories
    based on the available pharmaceutical product monograph headings.
    
    Returns a list of relevant section categories.
    """
    prompt = (
        "You are a clinical assistant AI that classifies drug-related user queries into predefined categories.\n\n"
        "These categories correspond to the official section headings in pharmaceutical product monographs. Each section addresses "
        "a different type of medical information (e.g., dosage, adverse reactions, contraindications, etc.).\n\n"
        "Your task is to identify **one or more** relevant section categories that the user's query falls under.\n\n"
        "**Instructions:**\n"
        "- Select all categories that are clearly relevant to the user's question and list each on a separate line.\n"
        "- DO NOT invent or modify the categories. Only choose from the list provided below.\n\n"
        f"**Available Categories:**\n{chr(10).join(SECTION_MAPPING.keys())}\n\n"
        f"**User Query:**\n{query}\n\n"
        "Return your answer as a list of category names (one per line):"
    )
    response = llm.invoke(prompt)
    content = response.content.strip()
    # Extract valid categories from response by matching against SECTION_MAPPING keys.
    categories = [line.strip() for line in content.split("\n") if line.strip() in SECTION_MAPPING]
    return categories

@tool
def identify_drug_source(query: str) -> str:
    """
    Identify the relevant drug source from the user's query. 
    
    Returns either "Metformin" or "Lipitor" based on which drug the query is about.
    """
    prompt = (
        "You are a clinical assistant AI that identifies the relevant drug for a given user query. "
        "Based on the query, answer with exactly one of the following: 'Metformin' or 'Lipitor'.\n\n"
        f"**User Query:**\n{query}\n\n"
        "Return your answer as exactly one word ('metformin', 'lipitor' or 'other'):"
    )
    response = llm.invoke(prompt)
    # In case the model returns additional text, take the first line.
    valid = {"metformin", "lipitor", "other"}
    drug = response.content.strip().lower().splitlines()[0]
    drug_source = drug if drug in valid else "other"
    return drug_source

@tool
def classify_intents(query: str) -> dict:
    """
    Run two sub-LLM calls:
      1. Classify the query into relevant section categories.
      2. Identify the drug source (Metformin or Lipitor).
      
    Returns a dictionary with:
      - 'intents': A list of relevant section categories.
      - 'drug_source': The identified drug source.
    """
    sections = classify_sections.invoke({'query': query})
    drug_source = identify_drug_source.invoke({'query': query})
    return {"intents": sections, "drug_source": drug_source.lower()}

@tool
def retrieve_relevant_chunks(query: str, intents: list, top_k_per_intent: int = 3, allowed_sources: str = None) -> list:
    """
    Retrieve context chunks from relevant sections based on classified intents,
    and also retrieve fallback chunks without any section filter.
    
    Applies allowed_sources filtering to both sectional and fallback queries.
    
    Parameters:
        query (str): The user's query.
        intents (list): List of section categories.
        top_k_per_intent (int): Number of top results to retrieve per intent.
        allowed_sources (str, optional): List of allowed sources to filter the results by.
        
    Returns:
        list: A list of unique context chunks.
    """
    all_chunks = []
    for intent in intents:
        sections = SECTION_MAPPING[intent]
        print(f"🔎 Retrieving for intent '{intent}' via sections {sections}")
        results = retrieve_top_k_sectional(query, k=top_k_per_intent, allowed_sections=sections, allowed_sources=allowed_sources)
        all_chunks.extend([r.payload for r in results])

    print("➕ Adding fallback top-k results (no section filter)...")
    fallback_results = retrieve_top_k(query, k=top_k_per_intent, filter_types=["text"], allowed_sources=allowed_sources)
    all_chunks.extend([r.payload for r in fallback_results])

    # Ensure chunks are unique based on their ID.
    unique_chunks = {chunk["chunk_id"]: chunk for chunk in all_chunks}.values()
    return list(unique_chunks)

@tool
def generate_citations(chunks: list) -> tuple:
    """
    Generate Vancouver-style numbered citations within the context and a reference list.
    Returns a tuple containing the final context text and a reference list.
    """
    numbered_chunks = []
    references = []

    for i, chunk in enumerate(chunks, start=1):
        text = chunk["text"]
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "?")
        section = chunk.get("section", "")
        url = chunk.get("url","")
        citation = f"[{i}]"
        c_id = chunk.get("chunk_id", "id")
        numbered_chunks.append(f"{text} {citation}")
        references.append(f"{citation} {source}.pdf => Section {section} => Page {page} => URL {url}")

    context_text = "\n\n".join(numbered_chunks)
    references_text = "\n".join(references)
    return context_text, references_text

@tool
def generate_answer(user_query: str, context: str, references: str) -> str:
    """
    Generate the final answer using the user query along with the retrieved context and citations.
    """
    prompt = f"""You are a knowledgeable clinical assistant with expertise in evidence-based medicine. 
    Using the comprehensive and trusted context provided below, generate an accurate, concise, and clear answer to the user's question. 
    The answer should include inline citations at the appropriate locations without a separate reference list.

    Question: {user_query}

    Context:
    {context}

    References:
    {references}

    Now, provide your final answer with correct inline Vancouver-style citations in exact location (ONLY ADD CITATION AND DON'T ADD REFERENCE): 
    """

    return llm.invoke(prompt).content



@tool
def plan_verification_questions(query: str, baseline: str) -> list:
    """
    Generate yes/no verification questions for fact-checking the baseline answer.
    """
    prompt = f"""You are a clinical assistant AI. Given the following:

    User Query: {query}
    Baseline Answer: {baseline}

    Generate a list of independent yes/no verification questions to validate each factual claim in the baseline answer.
    List each question on a new line."""
    response = llm.invoke(prompt)
    questions = [line.strip() for line in response.content.split("\n") if line.strip()]
    return questions


@tool
def execute_verifications(verification_questions: list) -> list:
    """
    Answer each verification question with 'yes' or 'no', verifying the factual accuracy.
    """
    answers = []
    for question in verification_questions:
        prompt = f"""You are a clinical assistant AI. Answer the following with a single word ('yes' or 'no') indicating whether the factual claim is correct:

        Verification Question: {question}
        Answer:"""
        response = llm.invoke(prompt)
        answers.append(response.content.strip())
    return answers


@tool
def generate_verified_answer(
    query: str,
    baseline: str,
    verif_questions: list,
    verif_answers: list,
    context: str,
    references: str
) -> str:
    """
    Given the original query, the baseline answer, verification questions/answers,
    the full context, and citation references, generate a factually accurate revised answer
    with inline Vancouver-style citations.
    """
    # Build a verification summary string
    verification_summary = ""
    for q, a in zip(verif_questions, verif_answers):
        verification_summary += f"\nVerification Question: {q}\nAnswer: {a}\n"

    prompt = f"""You are a clinical assistant AI with advanced evidence-based knowledge. Your task is to verify and correct the initial answer using the verification Q&A and trusted clinical context. 
    The answer should include inline citations at the appropriate locations without a separate reference list.

    User Query:
    {query}

    Baseline Answer:
    {baseline}

    Verification Summary:
    {verification_summary}

    Context:
    {context}

    References:
    {references}

    Now, provide your final verified answer with correct inline Vancouver-style citations in exact location (ONLY ADD CITATION AND DON'T ADD REFERENCE): 
    """

    return llm.invoke(prompt).content

