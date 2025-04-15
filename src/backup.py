# # src/rag_agent.py
# import os
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from retriever import (
#     retrieve_top_k,
#     retrieve_top_k_sectional,
#     SECTION_MAPPING,
# )

# # Load environment variables
# load_dotenv()

# # Initialize base LLM
# llm = ChatOpenAI(
#     model="gpt-4",
#     temperature=0,
#     api_key=os.getenv("OPENAI_API_KEY"),
#     model_kwargs={
#         "messages": [
#             {"role": "system", "content": "You are a clinical assistant. Only answer based on retrieved context. Always be truthful."}
#         ]
#     }
# )

# def classify_intent_and_sections(query: str):
#     intent_prompt = (
#         "You are a clinical assistant AI that classifies drug-related user queries into predefined categories.\n\n"
#         "These categories correspond to the official section headings in pharmaceutical product monographs. Each section addresses a different type of medical information (e.g., dosage, adverse reactions, contraindications, etc.).\n\n"
#         "Your task is to identify **one or more** relevant section categories that the user's query falls under.\n\n"
#         "**Instructions:**\n"
#         "- Select all categories that are clearly relevant to the user's question.\n"
#         "- DO NOT invent or modify the categories. Only choose from the list provided.\n"
#         "- If more than one category applies, list each one on a separate line.\n"
#         "- Your response must contain only the category names, exactly as listed — one per line.\n\n"
#         f"**Available Categories:**\n{chr(10).join(SECTION_MAPPING.keys())}\n\n"
#         f"**User Query:**\n{query}\n\n"
#         "Return your answer below:\n"
#     )
#     response = llm.invoke(intent_prompt)
#     return [key.strip() for key in response.content.split("\n") if key.strip() in SECTION_MAPPING]

# def generate_vancouver_citations(all_chunks: list[dict]) -> tuple[str, str]:
#     """Returns in-text context with Vancouver-style numbered citations and a reference list."""
#     numbered_chunks = []
#     references = []

#     for i, chunk in enumerate(all_chunks, start=1):
#         text = chunk["text"]
#         source = chunk.get("source", "Unknown")
#         page = chunk.get("page", "?")
#         chunk_id = chunk.get("chunk_id", "")

#         citation = f"[{i}]"
#         numbered_chunks.append(f"{text} {citation}")
#         references.append(f"{citation} {source}. Page {page}. ID: {chunk_id}")

#     return "\n\n".join(numbered_chunks), "\n".join(references)

# def section_aware_pipeline(user_query: str, top_k_per_intent: int = 3) -> str:
#     # Step 1: Classify into section-intents
#     intents = classify_intent_and_sections(user_query)
#     print("🎯 Intents:", intents)

#     # Step 2: Retrieve chunks from relevant sections
#     all_chunks = []
#     for intent in intents:
#         sections = SECTION_MAPPING[intent]
#         print(f"🔎 Retrieving for intent '{intent}' via sections {sections}")
#         results = retrieve_top_k_sectional(user_query, k=top_k_per_intent, allowed_sections=sections)
#         all_chunks.extend([r.payload for r in results])

#     # Step 3: Fallback top-k query (no section filter)
#     print("➕ Adding fallback top-k results (no section filter)...")
#     fallback_results = retrieve_top_k(user_query, k=top_k_per_intent, filter_types=["text"])
#     all_chunks.extend([r.payload for r in fallback_results])

#     # Step 4: Generate Vancouver-style citations
#     unique_chunks = {chunk["chunk_id"]: chunk for chunk in all_chunks}.values()
#     context, references = generate_vancouver_citations(list(unique_chunks))

#     # Step 5: Build prompt and invoke LLM
#     prompt = f"""You are a clinical assistant. Use the retrieved context below to answer the user's question. Be precise, truthful, and reference sources inline using the numbered citations.

#                 Question: {user_query}

#                 Context:
#                 {context}

#                 References:
#                 {references}
#                 """
#     return llm.invoke(prompt).content, references

# # Run query
# if __name__ == "__main__":
#     query = "What are the side effects of Lipitor in elderly patients?"
#     final_answer, references = section_aware_pipeline(query)
#     print("\n✅ Final Answer:")
#     print(final_answer)
#     print("\n📚 References:")
#     print(references)


# src/rag_agentic.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from retriever import (
    retrieve_top_k,
    retrieve_top_k_sectional,
    SECTION_MAPPING,
)

# Load environment variables
load_dotenv()

# Initialize the base LLM with a clinical assistant role.
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    model_kwargs={
        "messages": [
            {
                "role": "system",
                "content": "You are a clinical assistant. Only answer based on retrieved context. Always be truthful."
            }
        ]
    }
)

# Define a decorator to mark functions as agent workflow tools.
def tool(func):
    """Decorator marking a function as an agentic tool."""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@tool
def classify_intents(query: str) -> list:
    """
    Classify the user's query into one or more predefined pharmaceutical product monograph sections.
    Returns a list of relevant section categories.
    """
    intent_prompt = (
        "You are a clinical assistant AI that classifies drug-related user queries into predefined categories.\n\n"
        "These categories correspond to the official section headings in pharmaceutical product monographs. "
        "Each section addresses a different type of medical information (e.g., dosage, adverse reactions, contraindications, etc.).\n\n"
        "Your task is to identify **one or more** relevant section categories that the user's query falls under.\n\n"
        "**Instructions:**\n"
        "- Select all categories that are clearly relevant to the user's question.\n"
        "- DO NOT invent or modify the categories. Only choose from the list provided.\n"
        "- If more than one category applies, list each one on a separate line.\n"
        "- Your response must contain only the category names, exactly as listed — one per line.\n\n"
        f"**Available Categories:**\n{chr(10).join(SECTION_MAPPING.keys())}\n\n"
        f"**User Query:**\n{query}\n\n"
        "Return your answer below:\n"
    )
    response = llm.invoke(intent_prompt)
    # Only keep exact matching keys from SECTION_MAPPING
    return [key.strip() for key in response.content.split("\n") if key.strip() in SECTION_MAPPING]

@tool
def retrieve_relevant_chunks(query: str, intents: list, top_k_per_intent: int = 3) -> list:
    """
    Retrieve context chunks from relevant sections based on classified intents.
    Also retrieves fallback chunks without any section filter.
    """
    all_chunks = []
    for intent in intents:
        sections = SECTION_MAPPING[intent]
        print(f"🔎 Retrieving for intent '{intent}' via sections {sections}")
        results = retrieve_top_k_sectional(query, k=top_k_per_intent, allowed_sections=sections)
        all_chunks.extend([r.payload for r in results])

    print("➕ Adding fallback top-k results (no section filter)...")
    fallback_results = retrieve_top_k(query, k=top_k_per_intent, filter_types=["text"])
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
        chunk_id = chunk.get("chunk_id", "")
        citation = f"[{i}]"
        numbered_chunks.append(f"{text} {citation}")
        references.append(f"{citation} {source}. Page {page}. ID: {chunk_id}")

    context_text = "\n\n".join(numbered_chunks)
    references_text = "\n".join(references)
    return context_text, references_text

@tool
def generate_answer(user_query: str, context: str, references: str) -> str:
    """
    Generate the final answer using the user query along with the retrieved context and citations.
    """
    prompt = f"""You are a clinical assistant. Use the retrieved context below to answer the user's question. Be precise, truthful, and reference sources inline using the numbered citations.

Question: {user_query}

Context:
{context}

References:
{references}
"""
    return llm.invoke(prompt).content

def run_agent(user_query: str, top_k_per_intent: int = 3) -> tuple:
    """
    Run the complete agentic workflow:
        1. Classify query into relevant intents.
        2. Retrieve context chunks.
        3. Generate citations.
        4. Generate the final answer.
    Returns the final answer and reference list.
    """
    # Step 1: Classify the query into relevant section categories.
    intents = classify_intents(user_query)
    print("🎯 Classified Intents:", intents)

    # Step 2: Retrieve context chunks based on the classified intents.
    relevant_chunks = retrieve_relevant_chunks(user_query, intents, top_k_per_intent)

    # Step 3: Generate Vancouver-style citations.
    context, references = generate_citations(relevant_chunks)

    # Step 4: Generate the final answer using the retrieved context.
    final_answer = generate_answer(user_query, context, references)

    return final_answer, references

if __name__ == "__main__":
    # Run a sample query through the agentic workflow.
    query = "What are the side effects of Lipitor in elderly patients?"
    final_answer, references = run_agent(query)
    print("\n✅ Final Answer:\n", final_answer)
    print("\n📚 References:\n", references)



# # src/streamlit_app.py
# import streamlit as st
# import os
# from dotenv import load_dotenv
# from rag_agent import agent

# # Load environment variables
# load_dotenv()

# # Streamlit UI
# st.set_page_config(page_title="Clinical Agentic RAG", layout="centered")
# st.title("🧠 Clinical Question Answering")
# st.caption("Ask a question based on Lipitor or Metformin monographs.")

# user_query = st.text_area("Enter your question:")

# if st.button("Ask") and user_query.strip():
#     with st.spinner("Thinking..."):
#         try:
#             response = agent.invoke({"input": user_query})
#             st.success("Answer:")
#             st.write(response["output"])
#         except Exception as e:
#             st.error(f"❌ An error occurred: {e}")








# src/rag_agentic.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from retriever import (
    retrieve_top_k,
    retrieve_top_k_sectional,
    SECTION_MAPPING,
    retrieve_image_path
)

# Load environment variables
load_dotenv()

# Initialize the base LLM with a clinical assistant role.
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    model_kwargs={
        "messages": [
            {
                "role": "system",
                "content": "You are a clinical assistant. Only answer based on retrieved context. Always be truthful."
            }
        ]
    }
)

# # Define a decorator to mark functions as agent workflow tools.
# def tool(func):
#     """Decorator marking a function as an agentic tool."""
#     def wrapper(*args, **kwargs):
#         return func(*args, **kwargs)
#     return wrapper

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
    sections = classify_sections(query)
    drug_source = identify_drug_source(query)
    return {"intents": sections, "drug_source": drug_source.lower()}

@tool
def retrieve_relevant_chunks(query: str, intents: list, top_k_per_intent: int = 3, allowed_sources: list[str] = None) -> list:
    """
    Retrieve context chunks from relevant sections based on classified intents,
    and also retrieve fallback chunks without any section filter.
    
    Applies allowed_sources filtering to both sectional and fallback queries.
    
    Parameters:
        query (str): The user's query.
        intents (list): List of section categories.
        top_k_per_intent (int): Number of top results to retrieve per intent.
        allowed_sources (list[str], optional): List of allowed sources to filter the results by.
        
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
        # url = chunk["url"]
        citation = f"[{i}]"
        numbered_chunks.append(f"{text} {citation}")
        references.append(f"{citation} {source}. Page {page}. ")

    context_text = "\n\n".join(numbered_chunks)
    references_text = "\n".join(references)
    return context_text, references_text

@tool
def generate_answer(user_query: str, context: str, references: str) -> str:
    """
    Generate the final answer using the user query along with the retrieved context and citations.
    """
    prompt = f"""You are a clinical assistant. Use the retrieved context below to answer the user's question. Be precise, truthful, and reference sources inline using the numbered citations. Be sure to answer the user query.

            Question: {user_query}

            Context:
            {context}

            References:
            {references}
            """
    return llm.invoke(prompt).content

def run_agent(user_query: str, top_k_per_intent: int = 3) -> tuple:
    """
    Run the complete agentic workflow:
      1. Classify the query into relevant section categories and determine the drug source.
      2. Retrieve context chunks, filtering by the allowed drug source.
      3. Generate Vancouver-style citations.
      4. Modify the query to incorporate the drug source and generate the final answer.
      
    Returns the final answer and reference list.
    """
    # Step 1: Classify the query and determine the drug source.
    classification = classify_intents.invoke({"query": user_query})
    intents = classification.get("intents", [])
    drug_source = classification.get("drug_source", "Unknown")
    print("🎯 Classified Intents:", intents)
    print("💊 Identified Drug Source:", drug_source)

    # routing directly to LLM
    if drug_source == 'other':
        final_answer = generate_answer.invoke({
            "user_query": user_query,
            "context": "None",
            "references": "None"
        })
        return final_answer, None

    # Step 2: Retrieve context chunks using the allowed_sources filter.
    relevant_chunks = retrieve_relevant_chunks.invoke({
        "query": user_query,
        "intents": intents,
        "top_k_per_intent": top_k_per_intent,
        "allowed_sources": [drug_source]  # Ensure it's a list
    })

    # Step 3: Generate Vancouver-style citations from the retrieved chunks.
    context, references = generate_citations.invoke({"chunks": relevant_chunks})

    # Step 4: Modify the query to include the drug source and generate the final answer.
    modified_query = f"{user_query}\n(Note: The relevant drug is {drug_source})"
    final_answer = generate_answer.invoke({
        "user_query": modified_query,
        "context": context,
        "references": references
    })

    return final_answer, references


@tool
def plan_verification_questions(query: str, baseline: str) -> list:
    """
    Given the original query and the baseline answer, generate a list of verification questions 
    to check the factual claims in the baseline response.
    Returns a list of verification questions.
    """
    prompt = f"""You are a clinical assistant AI tasked with verifying factual claims in a response.
            Given the following:
            User Query: {query}

            Baseline Answer: {baseline}

            Generate a list of verification questions that could be used to fact-check the baseline answer.
            List each question on a new line without a numbering system.
            """
    response = llm.invoke(prompt)
    questions = [line.strip() for line in response.content.split("\n") if line.strip()]
    return questions

@tool
def execute_verifications(verification_questions: list) -> list:
    """
    Independently answer each verification question.
    Returns a list of verification answers corresponding to the questions.
    """
    answers = []
    for question in verification_questions:
        prompt = f"""You are a clinical assistant AI. Answer the following verification question with a concise factual response:

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

    prompt = f"""You are a clinical assistant AI. Your task is to verify and revise the initial answer
            using a set of verification questions/answers and trusted clinical context. Be precise, truthful, and reference sources inline using the numbered citations. 
            Be sure to answer the user query.

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

            Generate the final, verified answer below:
            """
    return llm.invoke(prompt).content


# ------ Modified run_agent incorporating Chain-of-Verification ------ #

def run_and_verify_agent(user_query: str, top_k_per_intent: int = 3) -> tuple:
    """
    Run the complete agentic workflow with Chain-of-Verification using .invoke() calls:
      1. Classify the query into relevant section categories and determine the drug source.
      2. Retrieve context chunks (via allowed_sources filtering).
      3. Generate Vancouver-style citations.
      4. Generate a baseline answer using the retrieved context.
      5. Plan verification questions for the baseline answer.
      6. Execute the verification questions.
      7. Generate the final verified answer by incorporating the verification results.
      
    Returns:
      final_verified_answer (str): The revised answer after verification.
      references (str): Vancouver-style citations.
    """
    # Step 1: Classify the query and determine the drug source.
    classification = classify_intents.invoke({"query": user_query})
    intents = classification.get("intents", [])
    drug_source = classification.get("drug_source", "unknown")
    print("🎯 Classified Intents:", intents)
    print("💊 Identified Drug Source:", drug_source)

    # Routing directly to LLM if drug_source == 'other'
    if drug_source == "other":
        baseline_answer = generate_answer.invoke({
            "user_query": user_query,
            "context": "None",
            "references": "None"
        })
        verif_questions = plan_verification_questions.invoke({
            "query": user_query,
            "baseline": baseline_answer
        })
        verif_answers = execute_verifications.invoke({
            "verification_questions": verif_questions
        })
        final_verified_answer = generate_verified_answer.invoke({
            "query": user_query,
            "baseline": baseline_answer,
            "verif_questions": verif_questions,
            "verif_answers": verif_answers,
            "context": "None",
            "references": "None"
        })
        return final_verified_answer, None

    # Step 2: Retrieve context chunks using the allowed_sources filter.
    # Wrap drug_source in a list if the retrieval expects a list.
    relevant_chunks = retrieve_relevant_chunks.invoke({
        "query": user_query,
        "intents": intents,
        "top_k_per_intent": top_k_per_intent,
        "allowed_sources": [drug_source]
    })
    
    # Step 3: Generate Vancouver-style citations from the retrieved chunks.
    context_tuple = generate_citations.invoke({"chunks": relevant_chunks})
    context, references = context_tuple

    # Step 4: Modify the query to include the drug source and generate the baseline answer.
    modified_query = f"{user_query}\n(Note: The relevant drug is {drug_source})"
    baseline_answer = generate_answer.invoke({
        "user_query": modified_query,
        "context": context,
        "references": references
    })

    # Step 5: Plan verification questions based on the baseline answer.
    verif_questions = plan_verification_questions.invoke({
        "query": user_query,
        "baseline": baseline_answer
    })

    # Step 6: Execute verification questions independently.
    verif_answers = execute_verifications.invoke({
        "verification_questions": verif_questions
    })

    # Step 7: Generate final verified answer incorporating verification results.
    final_verified_answer = generate_verified_answer.invoke({
        "query": user_query,
        "baseline": baseline_answer,
        "verif_questions": verif_questions,
        "verif_answers": verif_answers,
        "context": context,
        "references": references
    })

    return final_verified_answer, references

if __name__ == "__main__":
    # Run a sample query through the agentic workflow with Chain-of-Verification.
    query = "What are the side effects of Lipitor in elderly patients?"
    final_answer, references = run_and_verify_agent(query)
    print("\n✅ Final Verified Answer:\n", final_answer)
    print("\n📚 References:\n", references)



# # src/rag_agentic.py
# from tools import classify_intents,generate_answer, generate_citations, generate_verified_answer
# from tools import retrieve_image_path, retrieve_relevant_chunks, plan_verification_questions, execute_verifications

# def run_agent(user_query: str, top_k_per_intent: int = 3) -> tuple:
#     """
#     Run the complete agentic workflow:
#       1. Classify the query into relevant section categories and determine the drug source.
#       2. Retrieve context chunks, filtering by the allowed drug source.
#       3. Generate Vancouver-style citations.
#       4. Modify the query to incorporate the drug source and generate the final answer.
      
#     Returns the final answer and reference list.
#     """
#     # Step 1: Classify the query and determine the drug source.
#     classification = classify_intents.invoke({"query": user_query})
#     intents = classification.get("intents", [])
#     drug_source = classification.get("drug_source", "unknown")
#     print("🎯 Classified Intents:", intents)
#     print("💊 Identified Drug Source:", drug_source)

#     # If the drug source is 'other', route directly to generate an answer.
#     if drug_source == "other":
#         final_answer = generate_answer.invoke({
#             "user_query": user_query,
#             "context": "None",
#             "references": "None"
#         })
#         return final_answer, None

#     # Step 2: Retrieve context chunks using the allowed_sources filter.
#     relevant_chunks = retrieve_relevant_chunks.invoke({
#         "query": user_query,
#         "intents": intents,
#         "top_k_per_intent": top_k_per_intent,
#         "allowed_sources": drug_source
#     })

#     # Step 3: Generate Vancouver-style citations from the retrieved chunks.
#     context_tuple = generate_citations.invoke({"chunks": relevant_chunks})
#     context, references = context_tuple

#     # Step 4: Modify the query to include the drug source and generate the final answer.
#     modified_query = f"{user_query}\n(Note: The relevant drug is {drug_source})"
#     final_answer = generate_answer.invoke({
#         "user_query": modified_query,
#         "context": context,
#         "references": references
#     })
    
#     return final_answer, references



# # ------ Modified run_agent incorporating Chain-of-Verification ------ #

# def run_and_verify_agent(user_query: str, top_k_per_intent: int = 3) -> tuple:
#     """
#     Run the complete agentic workflow with Chain-of-Verification:
#       1. Classify the query into relevant section categories and determine the drug source.
#       2. Retrieve context chunks, filtering by the allowed drug source.
#       3. Generate Vancouver-style citations.
#       4. Generate a baseline answer using the retrieved context.
#       5. Plan verification questions for the baseline answer.
#       6. Execute the verification questions.
#       7. Generate the final verified answer by incorporating the verification results.
      
#     Returns:
#       final_verified_answer (str): The revised answer after verification.
#       references (str): Vancouver-style citations.
#     """
#     # Step 1: Classify the query and determine the drug source.
#     classification = classify_intents(user_query)
#     intents = classification.get("intents", [])
#     drug_source = classification.get("drug_source", None)
#     print("🎯 Classified Intents:", intents)
#     print("💊 Identified Drug Source:", drug_source)

#         # routing directly to LLM
#     if drug_source == 'other':    
#         baseline_answer = generate_answer.invoke({
#             "user_query": user_query,
#             "context": "None",
#             "references": "None"
#         })
#         verif_questions = plan_verification_questions.invoke({'query': user_query, 'baseline': baseline_answer})
#         verif_answers = execute_verifications.invoke({'verification_questions': verif_questions})
#         final_verified_answer = generate_verified_answer.invoke({
#             "query": user_query,
#             "baseline": baseline_answer,
#             "verif_questions": verif_questions,
#             "verif_answers": verif_answers,
#             "context": "None",
#             "references": "None"
#             })
        
#         yes_count = sum(1 for answer in verif_answers if answer.lower() in ["yes", "true", "correct"])
#         total = len(verif_answers)
#         evaluation_ratio = f"{yes_count}/{total}"  # or as a percentage
#         return final_verified_answer, None, evaluation_ratio
    
#     # Step 2: Retrieve context chunks using allowed source filtering.
#     relevant_chunks = retrieve_relevant_chunks.invoke({
#         "query": user_query,
#         "intents": intents,
#         "top_k_per_intent": top_k_per_intent,
#         "allowed_sources": drug_source
#     })

    
#     # Step 3: Generate Vancouver-style citations from the retrieved chunks.
#     context, references = generate_citations.invoke({'chunks': relevant_chunks})
    
#     # Step 4: Generate the baseline answer.
#     modified_query = f"{user_query}\n(Note: The relevant drug is {drug_source})"
#     baseline_answer = generate_answer.invoke({'user_query': modified_query, 'context': context, 'references': references})
    
#     # Step 5: Plan verification questions based on the baseline answer.
#     verif_questions = plan_verification_questions.invoke({'query': user_query, 'baseline': baseline_answer})
    
#     # Step 6: Execute verification questions independently.
#     verif_answers = execute_verifications.invoke({'verification_questions': verif_questions})
#     print("verify answers: ")
#     print(verif_answers)
#     # Step 7: Generate final verified answer incorporating verification results.
#     final_verified_answer = generate_verified_answer.invoke({
#         "query": user_query,
#         "baseline": baseline_answer,
#         "verif_questions": verif_questions,
#         "verif_answers": verif_answers,
#         "context": context,
#         "references": references
#     })
#     yes_count = sum(1 for answer in verif_answers if answer.lower() in ["yes", "true", "correct"])
#     total = len(verif_answers)
#     evaluation_ratio = f"{yes_count}/{total}"
    
#     return final_verified_answer, references, evaluation_ratio

# def run_image_agent(query: str, top_k: int = 1) -> tuple:
#     """
#     Run the image agent workflow:
#       Given a user query, retrieve the top image path and its caption.
      
#     Returns a tuple containing:
#       - image_path (str): The file path of the retrieved image.
#       - caption (str): The caption associated with the image.
#     """
#     print("Inside run_image_agent: ")
#     image_path, caption = retrieve_image_path.invoke({
#         "query": query,
#         "top_k": top_k
#     })
#     print("Path: ", image_path, " caption ", caption)
#     return image_path, caption

# if __name__ == "__main__":
#     # Run a sample query through the agentic workflow with Chain-of-Verification.
#     query = "What are the side effects of Lipitor in elderly patients?"
#     final_answer, references, eval_ratio = run_and_verify_agent(query)
#     print("\n✅ Final Verified Answer:\n", final_answer)
#     print("\n📚 References:\n", references)
#     print("\n Eval Ratio: ", eval_ratio)


#     final_answer, references = run_agent(query)
#     print("\n✅ Final Answer:\n", final_answer)
#     print("\n📚 References:\n", references)

#     x, y = run_image_agent(query)
#     print("X: ", x, " Y: ", y)

# # src/rag_agentic.py
# from tools import (
#     classify_intents,
#     generate_answer,
#     generate_citations,
#     generate_verified_answer,
#     retrieve_image_path,
#     retrieve_relevant_chunks,
#     plan_verification_questions,
#     execute_verifications,
# )

# def run_agent(user_query: str, top_k_per_intent: int = 3) -> tuple:
#     """
#     Execute the full workflow:
#       - Classify query & identify drug source.
#       - Retrieve context chunks.
#       - Generate citations.
#       - Produce final answer including drug source.
      
#     Returns:
#       tuple: (final answer, references)
#     """
#     # Classify query and determine drug source.
#     classification = classify_intents.invoke({"query": user_query})
#     intents = classification.get("intents", [])
#     drug_source = classification.get("drug_source", "unknown")
#     print("🎯 Classified Intents:", intents)
#     print("💊 Identified Drug Source:", drug_source)

#     # For 'other', generate answer directly.
#     if drug_source == "other":
#         final_answer = generate_answer.invoke({
#             "user_query": user_query,
#             "context": "None",
#             "references": "None"
#         })
#         return final_answer, None

#     # Retrieve context and generate citations.
#     relevant_chunks = retrieve_relevant_chunks.invoke({
#         "query": user_query,
#         "intents": intents,
#         "top_k_per_intent": top_k_per_intent,
#         "allowed_sources": drug_source
#     })
#     context, references = generate_citations.invoke({"chunks": relevant_chunks})

#     # Form modified query with drug source and generate final answer.
#     modified_query = f"{user_query}\n(Note: The relevant drug is {drug_source})"
#     final_answer = generate_answer.invoke({
#         "user_query": modified_query,
#         "context": context,
#         "references": references
#     })
    
#     return final_answer, references

# def run_and_verify_agent(user_query: str, top_k_per_intent: int = 3) -> tuple:
#     """
#     Execute workflow with verification:
#       - Classify query & determine drug source.
#       - Retrieve context and generate baseline answer with citations.
#       - Plan & execute verification questions.
#       - Generate final verified answer.
      
#     Returns:
#       tuple: (final verified answer, references, evaluation ratio)
#     """
#     classification = classify_intents(user_query)
#     intents = classification.get("intents", [])
#     drug_source = classification.get("drug_source", None)
#     print("🎯 Classified Intents:", intents)
#     print("💊 Identified Drug Source:", drug_source)

#     if drug_source == "other":
#         baseline_answer = generate_answer.invoke({
#             "user_query": user_query,
#             "context": "None",
#             "references": "None"
#         })
#         verif_questions = plan_verification_questions.invoke({
#             "query": user_query,
#             "baseline": baseline_answer
#         })
#         verif_answers = execute_verifications.invoke({
#             "verification_questions": verif_questions
#         })
#         final_verified_answer = generate_verified_answer.invoke({
#             "query": user_query,
#             "baseline": baseline_answer,
#             "verif_questions": verif_questions,
#             "verif_answers": verif_answers,
#             "context": "None",
#             "references": "None"
#         })
#         yes_count = sum(1 for answer in verif_answers if answer.lower() in ["yes", "true", "correct"])
#         total = len(verif_answers)
#         evaluation_ratio = f"{yes_count}/{total}"
#         return final_verified_answer, None, evaluation_ratio

#     # Retrieve context and generate citations.
#     relevant_chunks = retrieve_relevant_chunks.invoke({
#         "query": user_query,
#         "intents": intents,
#         "top_k_per_intent": top_k_per_intent,
#         "allowed_sources": drug_source
#     })
#     context, references = generate_citations.invoke({"chunks": relevant_chunks})
    
#     # Generate baseline answer.
#     modified_query = f"{user_query}\n(Note: The relevant drug is {drug_source})"
#     baseline_answer = generate_answer.invoke({
#         "user_query": modified_query,
#         "context": context,
#         "references": references
#     })
    
#     # Plan and execute verifications.
#     verif_questions = plan_verification_questions.invoke({
#         "query": user_query,
#         "baseline": baseline_answer
#     })
#     verif_answers = execute_verifications.invoke({
#         "verification_questions": verif_questions
#     })
#     print("verify answers:")
#     print(verif_answers)
    
#     # Generate final verified answer.
#     final_verified_answer = generate_verified_answer.invoke({
#         "query": user_query,
#         "baseline": baseline_answer,
#         "verif_questions": verif_questions,
#         "verif_answers": verif_answers,
#         "context": context,
#         "references": references
#     })
#     yes_count = sum(1 for answer in verif_answers if answer.lower() in ["yes", "true", "correct"])
#     total = len(verif_answers)
#     evaluation_ratio = f"{yes_count}/{total}"
    
#     return final_verified_answer, references, evaluation_ratio

# def run_image_agent(query: str, top_k: int = 1) -> tuple:
#     """
#     Retrieve the top image path and caption for a query.
    
#     Returns:
#       tuple: (image_path, caption)
#     """
#     print("Inside run_image_agent:")
#     image_path, caption = retrieve_image_path.invoke({
#         "query": query,
#         "top_k": top_k
#     })
#     print("Path:", image_path, "caption:", caption)
#     return image_path, caption

# if __name__ == "__main__":
#     query = "What are the side effects of Lipitor in elderly patients?"
    
#     final_answer, references, eval_ratio = run_and_verify_agent(query)
#     print("\n✅ Final Verified Answer:\n", final_answer)
#     print("\n📚 References:\n", references)
#     print("\nEval Ratio:", eval_ratio)

#     final_answer, references = run_agent(query)
#     print("\n✅ Final Answer:\n", final_answer)
#     print("\n📚 References:\n", references)

#     x, y = run_image_agent(query)
#     print("X:", x, "Y:", y)
